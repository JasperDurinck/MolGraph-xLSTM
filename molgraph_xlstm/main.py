import os

import time
import json
import warnings
from tqdm import tqdm
from copy import deepcopy
from typing import Set, Callable, Any

import pandas as pd
import numpy as np
from transformers import optimization

import torch
from torch import Tensor
from torch.nn import Module
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch_geometric.data import DataLoader, Data

from .utils.parsing import parse_args_dict
from .utils.evaluate import Evaluator
from .utils.load_dataset import process_df, PygOurDataset
from .utils.util import AverageMeter, set_optimizer, calmean
from .loss.loss_scl_cls import SupConLoss
from .loss.loss_scl_reg import SupConLossReg
from .models.deepgcn import GraphxLSTM
from .models.module import MLP, MLPMoE

warnings.filterwarnings("ignore")

class ContrastiveEncode(torch.nn.Module):
    """
    A simple feature encoder for contrastive learning tasks.

    This module applies a two-layer MLP with ReLU activation in between.
    It's useful for encoding input features (e.g., molecular embeddings) 
    into a latent space suitable for contrastive loss.

    Args:
        dim_feat (int): Dimensionality of input and output features.
    """
    def __init__(self, dim_feat: int):
        super().__init__()
        self.encode = torch.nn.Sequential(
                torch.nn.Linear(dim_feat, dim_feat),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(dim_feat, dim_feat)
        )

    def forward(self, x):
        x = self.encode(x)
        return x

class MolPropertyPrediction(torch.nn.Module):
    """
    Molecular Property Prediction Model using Multi-View Contrastive Feature Encoders.

    This model processes molecular inputs using a backbone encoder (e.g., LSTM-based)
    and extracts multiple levels of features: atom-level, functional group-level,
    graph-level, and a final aggregated representation. Each of these feature sets
    is passed through a separate contrastive encoder (a small MLP) to normalize and
    embed them into a latent space.

    The model then applies expert-based classifiers (Mixture of Experts) on each view,
    supporting multi-task property prediction. During training, an aggregate contrastive
    loss is computed from each of the views to improve representation learning.

    Args:
        molxlstm (torch.nn.Module): Backbone model that encodes molecular input into
            multiple hierarchical representations (atom, functional group, graph, joint).
        args (Any): Configuration object or namespace containing the following attributes:
            - num_dim (int): Base embedding dimensionality.
            - power (int): Expansion factor for dimensionality (final = num_dim * power).
            - num_tasks (int): Number of prediction tasks.
            - num_experts (int): Number of expert networks per classifier.
            - num_heads (int): Number of heads for attention or routing in MoE.

    Inputs:
        input_molecule (torch.Tensor): Molecular graph data, format depends on molxlstm.
        phase (str): Either "train" or "eval", controls whether to return loss.

    Returns:
        If phase == "train":
            Tuple[
                output_final, output_atom, output_fg, output_gnn,
                atom_feat_norm, fg_feat_norm, graph_feat_norm, f_out_norm,
                loss_auc
            ]
        Else:
            Tuple[
                output_final, output_atom, output_fg, output_gnn,
                atom_feat_norm, fg_feat_norm, graph_feat_norm, f_out_norm
            ]
    """
     
    def __init__(self, molxlstm: Module, args):
        super(MolPropertyPrediction, self).__init__()

        self.molxlstm = molxlstm

        self.dropout = torch.nn.Dropout(0.5)

        emb_dim = args['num_dim'] * args['power']
        self.enc_joint = ContrastiveEncode(emb_dim)
        self.enc_joint_1 = ContrastiveEncode(emb_dim)
        self.enc_joint_2 = ContrastiveEncode(emb_dim)
        self.enc_joint_3 = ContrastiveEncode(emb_dim)

        classifier_num = 11
        self.classifier = torch.nn.ModuleList()
        self.classifier.extend(
            MLPMoE(input_feat = args['num_dim'] * args['power'], dim_feat = args['num_dim'] * args['power'], 
            num_tasks=args['num_tasks'],
            num_experts = args['num_experts'], 
            num_heads = args['num_heads']) for _ in range(classifier_num)
            )

    def forward(self, input_molecule: Tensor, phase: str = "train"):
        f_out, atom_feat, fg_feat, graph_feat = self.molxlstm(input_molecule)

        atom_feat_norm = F.normalize(self.enc_joint_1(atom_feat), 1)
        fg_feat_norm = F.normalize(self.enc_joint_2(fg_feat), 1)
        graph_feat_norm = F.normalize(self.enc_joint_3(graph_feat), 1)
        f_out_norm = F.normalize(self.enc_joint(f_out), 1)

        output_atom, loss_atom = self.classifier[6](atom_feat.unsqueeze(1))
        output_fg, loss_fg = self.classifier[7](fg_feat.unsqueeze(1))
        output_gnn, loss_gnn = self.classifier[8](graph_feat.unsqueeze(1))
        output_final, loss_out = self.classifier[9](f_out.unsqueeze(1))
        loss_auc = (loss_atom + loss_fg + loss_out + loss_gnn)

        if phase == "train":
            return (
                output_final,
                output_atom,
                output_fg,
                output_gnn,
                atom_feat_norm,
                fg_feat_norm,
                graph_feat_norm,
                f_out_norm,
                loss_auc
            )
        else:
            return (
                output_final,
                output_atom,
                output_fg,
                output_gnn,
                atom_feat_norm,
                fg_feat_norm,
                graph_feat_norm,
                f_out_norm
            )

def set_seed(num_seed):
    np.random.seed(num_seed)
    torch.manual_seed(num_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(num_seed)
        torch.cuda.manual_seed_all(num_seed) 

def load_data(args: Any) -> Set[Data]:
    """Load dataset from args['datas_dir'].

    Args:
        opt (Any): Parsed arguments.
        dataname (str): The folder name of the dataset.

    Returns:
        Set[Data]: train/validation/test sets.
    """
    data = torch.load(args['data_file']) #holds df, train_idx, ... in a dict
    
    return data

def set_loss_fn(args):
    if args['classification']:
        criterion_task = torch.nn.BCEWithLogitsLoss()
    else:
        criterion_task = torch.nn.MSELoss()
    
    return criterion_task

def set_model(args: Any):
    """Initialization of the model and loss functions.

    Args:
        args (Any): Parsed arguments.

    Returns:
        Return the model and the loss functions.
    """

    molgraph_xlstm = GraphxLSTM(args)
    model = MolPropertyPrediction(molgraph_xlstm, args)

    if torch.cuda.is_available():
        model = model.cuda()
        cudnn.benchmark = False

    return model

def load_weights(model, args):
    """
    Load model weights from a checkpoint file.

    Args:
        model (torch.nn.Module): The model to load weights into.
        args (Namespace): Should contain 'checkpoint_path' attribute.
    """
    checkpoint = torch.load(args['checkpoint_path'], map_location='cpu')  # or 'cuda' if needed
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # In case the checkpoint keys are prefixed (e.g., 'module.' from DataParallel)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v  # remove 'module.' prefix
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=False)
    print("Weights loaded successfully!")
    return model 

def _train(
    train_loader: Any,
    model: torch.nn.Sequential,
    criterion_task: Callable,
    optimizer: Optimizer,
    scheduler: Any,
    args: Any,
    mu: int = 0,
    std: int = 0,
    dynamic_t: int = 0, 
    max_dist: int = 0,
    epoch: int = 0
):
    """One epoch training.

    Args:
        train_dataset (Set[Data]): Train set.
        model (torch.nn.Sequential): Model
        criterion_task (Callable): Task loss function
        optimizer (Optimizer): Optimizer
        opt (Any): Parsed arguments
        mu (int, optional): Mean value of the train set for the regression task. Defaults to 0.
        std (int, optional): Standard deviation of the train set for the regression task.
            Defaults to 0.

    Returns:
        Losses.
    """
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses_task = AverageMeter()
    losses_scl = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for _, batch in enumerate(tqdm(train_loader, desc="Iteration")):
        batch = batch.to("cuda")
        data_time.update(time.time() - end)

        bsz = batch.y.shape[0]
        if not args['classification']:
            labels = (batch.y - mu) / std
        else:
            labels = batch.y
        
        # compute loss
        (
            output_final,
            output_atom,
            output_fg,
            output_gnn,
            atom_feat,
            fg_feat,
            graph_feat,
            f_out,
            loss_auc
        ) = model(batch, 'train')


        if (args['classification']):
            criterion_cl = SupConLoss()
        else:
            criterion_cl = SupConLossReg(gamma1 = 1, gamma2 = 0)

        loss_cl_tmp = 0

        features_graph_1 = torch.cat([f_out.unsqueeze(1), f_out.unsqueeze(1)], dim=1)
        features_graph_2 = torch.cat([atom_feat.unsqueeze(1), atom_feat.unsqueeze(1)], dim=1)
        features_graph_3 = torch.cat([fg_feat.unsqueeze(1), fg_feat.unsqueeze(1)], dim=1)
        features_graph_4 = torch.cat([graph_feat.unsqueeze(1), graph_feat.unsqueeze(1)], dim=1)

        num_labels = 0
        loss_task_tmp_final = 0
        loss_task_tmp_atom = 0
        loss_task_tmp_fg = 0
        loss_task_tmp_gnn = 0

        for i in range(labels.shape[1]):
            is_labeled = batch.y[:, i] == batch.y[:, i]

            loss_task_final = criterion_task(
                    output_final[is_labeled, i].squeeze(), labels[is_labeled, i].squeeze()
            )

            loss_task_atom = criterion_task(
                    output_atom[is_labeled, i].squeeze(), labels[is_labeled, i].squeeze()
            )

            loss_task_fg = criterion_task(
                    output_fg[is_labeled, i].squeeze(), labels[is_labeled, i].squeeze()
            )

            loss_task_gnn = criterion_task(
                    output_gnn[is_labeled, i].squeeze(), labels[is_labeled, i].squeeze()
            )

            loss_task_tmp_final = loss_task_tmp_final + loss_task_final
            loss_task_tmp_atom = loss_task_tmp_atom + loss_task_atom
            loss_task_tmp_fg = loss_task_tmp_fg + loss_task_fg
            loss_task_tmp_gnn = loss_task_tmp_gnn + loss_task_gnn

            if labels[is_labeled, i].sum() == 0:
                continue

            num_labels = num_labels + 1

            if args['classification']:
                loss_cl_tmp = (loss_cl_tmp +
                               (criterion_cl(features_graph_1[is_labeled, :], labels[is_labeled, i].squeeze()) +
                                criterion_cl(features_graph_2[is_labeled, :], labels[is_labeled, i].squeeze()) +
                                criterion_cl(features_graph_3[is_labeled, :], labels[is_labeled, i].squeeze()) +
                                criterion_cl(features_graph_4[is_labeled, :], labels[is_labeled, i].squeeze())))
            else:
                loss_cl_tmp = (loss_cl_tmp +
                               criterion_cl(dynamic_t, max_dist, features_graph_1[is_labeled, :], labels[is_labeled, i].squeeze()) +
                               criterion_cl(dynamic_t, max_dist, features_graph_2[is_labeled, :], labels[is_labeled, i].squeeze()) +
                               criterion_cl(dynamic_t, max_dist, features_graph_3[is_labeled, :], labels[is_labeled, i].squeeze()) +
                               criterion_cl(dynamic_t, max_dist, features_graph_4[is_labeled, :], labels[is_labeled, i].squeeze()))

        if num_labels==0:
            loss_cl = loss_cl_tmp / labels.shape[1]
        else:
            loss_cl = loss_cl_tmp / num_labels

        loss_task = (loss_task_tmp_final + loss_task_tmp_atom + loss_task_tmp_fg + loss_task_tmp_gnn) / labels.shape[1]

        if args['classification']:
            loss = loss_task + loss_cl + loss_auc
        else:
            loss = loss_task + loss_cl + loss_auc

        # update metric
        losses_task.update(loss_task.item(), bsz)
        losses_scl.update(loss_cl.item(), bsz)
        losses.update(loss.item(), bsz)

        optimizer[0].zero_grad()
        loss.backward()
        optimizer[0].step()
        scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
    return losses_task.avg, losses_scl.avg, losses.avg

def _validation(
    dataset: Set[Data],
    model: torch.nn.Sequential,
    args: Any,
    mu: int = 0,
    std: int = 0,
):
    """Calculate performance metrics.

    Args:
        dataset (Set[Data]): A dataset.
        model (torch.nn.Sequential): Model.
        opt (Any): Parsed arguments.
        mu (int, optional): Mean value of the train set for the regression task.
            Defaults to 0.
        std (int, optional): Standard deviation of the train set for the regression task.
            Defaults to 0.
        save_feature (int, optional): Whether save the learned features or not.
            Defaults to 0.

    Returns:
        auroc or rmse value.
    """
    model.eval()

    if args['classification']:
        evaluator = Evaluator(name=args['dataname'], num_tasks=args['num_tasks'], eval_metric=args.get('score_metric', 'rocauc'))
    else:
        evaluator = Evaluator(name=args['dataname'], num_tasks=args['num_tasks'], eval_metric=args.get('score_metric', 'rmse'))
    data_loader = DataLoader(
        dataset, batch_size=args['batch_size'], shuffle=False, follow_batch = ['fg_x', 'atom2fg_list']
    )

    with torch.no_grad():
        y_true = []
        y_pred = []
        for _, batch in enumerate(tqdm(data_loader, desc="Iteration")):
            batch = batch.to("cuda")
            (
                output_final,
                output_atom,
                output_fg,
                output_gnn,
                atom_feat,
                fg_feat, 
                graph_feat,
                f_out
            ) = model(batch, "valid")

            if not args['classification']:
                output = (output_final) * std + mu

            if args['classification']:
                sigmoid = torch.nn.Sigmoid()
                output = sigmoid(output_final)

            y_true.append(batch.y.detach().cpu())
            y_pred.append(output.detach().cpu())

        y_true = torch.cat(y_true, dim=0).squeeze().unsqueeze(1).numpy()
        if args['num_tasks'] > 1:
            y_pred = np.concatenate(y_pred)
            input_dict = {"y_true": y_true.squeeze(), "y_pred": y_pred.squeeze()}
        else:
            y_pred = np.expand_dims(np.concatenate(y_pred), 1)
            input_dict = {
                "y_true": np.expand_dims(y_true.squeeze(), 1),
                "y_pred": np.expand_dims(y_pred.squeeze(), 1),
            }

        if args['classification']:
            eval_result = evaluator.eval(input_dict)[args.get('score_metric', 'rocauc')]
        else:
            eval_result = evaluator.eval(input_dict)[args.get('score_metric', 'rmse')]

    return y_true, y_pred, eval_result

def compute_z_scaling(y, args):
    """Compute Z-scaling (mean and std) for regression. Optionally save/load scaling parameters.

    Args:
        data_list (list): The list of Data objects containing features and targets.
        args (dict): Parsed arguments, containing 'classification' and other settings.
        save_dir (str, optional): Directory to save/load the scaling parameters. Defaults to None.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, float, float]: mu, std, dynamic_t, max_dist
    """
    if args['classification']:
        # For classification, no scaling (i.e., using defaults)
        mu, std, dynamic_t, max_dist = 0, 0, 0, 0
    else:
        # Check if scaling parameters already exist
        if args['re_calc_z_scale_settings'] is False and os.path.exists(args['z_scale_setting_save_path'] or y is None):
            # Load previously saved scaling parameters
            scaling_params = torch.load(args['z_scale_setting_save_path'])
            mu, std, dynamic_t, max_dist = scaling_params['mu'], scaling_params['std'], scaling_params['dynamic_t'], scaling_params['max_dist']
            print("Loaded saved scaling parameters.")
        elif args['re_calc_z_scale_settings'] is True or y is not None:
            # Compute scaling parameters (mean and std) for regression
            mu, std, dynamic_t, max_dist = calmean(y)
            
            # Save the scaling parameters if needed
            if args['z_scale_setting_save_path']:
                torch.save({
                    'mu': mu,
                    'std': std,
                    'dynamic_t': dynamic_t,
                    'max_dist': max_dist
                }, args['z_scale_setting_save_path'])
                print(f"Saved scaling parameters to {args['z_scale_setting_save_path']}")
        else:
            raise ValueError('Either load the data or have re_calc set True')
    
    return mu, std, dynamic_t, max_dist

def get_data_subset(data, subsets=['train_idx']):
    """
    data must be dict with X, y, train_idx, val_idx, test_idx 
    """
    # Combine indices for training
    train_indices = np.array([], dtype=int)
    for key in subsets:
        if key in data and data[key] is not None:
            train_indices = np.concatenate([train_indices, np.array(data[key])])

    # Input features
    if isinstance(data['x'], list):
        x = [data['x'][i] for i in train_indices]
    elif isinstance(data['x'], np.ndarray):
        x = data['x'][train_indices]

    # Targets
    if isinstance(data['y'], list):
        y = [data['y'][i] for i in train_indices]
    elif isinstance(data['y'], np.ndarray):
        y = data['y'][train_indices]
    
    return x, y

def make_dataloader(data_list, args):
    return PygOurDataset(data_list, args['smiles_col'])

def process_data(data, args):

    if args['processed_data_path'] is not None and os.path.exists(args['processed_data_path']):
        data = torch.load(args['processed_data_path'])
        return data

    data_ = process_df( 
        df=data['df'], 
        target_cols=args['target_cols'], 
        smiles_col=args.get('smiles_col', 'smiles'), 
        max_len=args['max_len_tokenizer']
    )
    
    data = {**data, **data_}
    
    if args['processed_data_path'] is not None:
        torch.save(data, args['processed_data_path'])
    
    return data

def fit_model(data, args):

    

    # Make datasets
    x_train, y_train = get_data_subset(data, subsets=args.get('fit_train_subset', ['train_idx']))
    x_val, y_val = get_data_subset(data, subsets=args.get('fit_val_subset', ['val_idx']))
    train_dataset, val_dataset = make_dataloader(x_train, args), make_dataloader(x_val, args)

    # Compute z-scale
    mu, std, dynamic_t, max_dist = compute_z_scaling(y_train, args)

    # Build model and criterion
    model, criterion_task = set_model(args), set_loss_fn(args)

    # Build optimizer
    optimizer = set_optimizer(args['learning_rate'], args['weight_decay'], model)

    train_loader = DataLoader(train_dataset, batch_size=args['batch_size'], drop_last=True, shuffle=True, follow_batch=['fg_x', 'atom2fg_list'])
    num_training_steps = len(train_loader) * args['epochs']
    num_warmup_steps = int(num_training_steps * 0.1)
    scheduler = optimization.get_linear_schedule_with_warmup(optimizer[0], num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

    # Track best model
    best_model = None
    best_metric_score = float('inf')  # Assuming we're minimizing the metric (e.g., RMSE)
    
    # Training routine
    for epoch in range(args['epochs']):
        torch.cuda.empty_cache()
        
        # Train for one epoch
        time1 = time.time()
        loss_task, loss_scl, loss = _train(
            train_loader,
            model,
            criterion_task,
            optimizer,
            scheduler,
            args,
            mu,
            std,
            dynamic_t,
            max_dist,
            epoch
        )
        time2 = time.time()
        print("epoch {}, total time {:.2f}".format(epoch, time2 - time1))

        val_y_true, val_y_pred, metric_score = _validation(val_dataset, model, args, mu, std)

        if args['classification']:
            metric_name = args.get('score_metric', 'rocauc')
        else:
            metric_name = args.get('score_metric', 'rmse')

        print("val {}:{}".format(metric_name, metric_score))

        # If tracking best model and the current metric score is better (lower)
        if args.get('track_best_model', False):
            if metric_score > best_metric_score:
                best_metric_score = metric_score
                best_model = deepcopy(model)  # Save a deep copy of the model with the best score

    # At the end, use the best model if tracked, otherwise just the last model
    if args.get('track_best_model', False) and best_model is not None:
        print("Using best model for final validation.")
        model = best_model  # Use the best model for final validation
        val_y_true, val_y_pred, metric_score = _validation(val_dataset, model, args, mu, std)
        print("Final validation {} with best model: {}".format(metric_name, metric_score))

    # Save the model (best model if tracked, otherwise last model)
    if args['checkpoint_path']:
        torch.save(model.state_dict(), args['checkpoint_path'])

    # Check if classification task or regression task
    if args['classification']:
        # If there are multiple labels or tasks, reshape the arrays to flatten them.
        if len(val_y_pred.shape) > 2:  # If multi-task or multi-label
            val_y_pred = val_y_pred.reshape(val_y_pred.shape[0], -1)
            val_y_true = val_y_true.reshape(val_y_true.shape[0], -1)

        # Now, val_y_pred and val_y_true should be of shape (633, num_labels) or (633, num_tasks * num_labels)
        return pd.DataFrame({
            'pred': list(val_y_pred.flatten()),  # Flatten the predictions if needed
            'known': list(val_y_true.flatten())  # Flatten the true values if needed
        })

def test_model(data, args):
    print(data.keys())
    x_test, y_test = get_data_subset(data, subsets=args.get('fit_test_subset', ['test_idx'])) 
    test_dataset = make_dataloader(x_test, args)

    #compute z-scale
    mu, std, dynamic_t, max_dist = compute_z_scaling(None, args)

    # build model and criterion
    model = set_model(args)
    model = load_weights(model, args)

    y_true, y_pred, test_acc = _validation(test_dataset, model.cuda(), args, mu, std)

    if args['classification']:
        metric_name = args.get('score_metric', 'rocauc')
    else:
        metric_name = args.get('score_metric', 'rmse')

    print("test {}:{}".format(metric_name, test_acc))

    # Check if classification task or regression task
    if args['classification']:
        # If there are multiple labels or tasks, reshape the arrays to flatten them.
        if len(y_pred.shape) > 2:  # If multi-task or multi-label
            y_pred = y_pred.reshape(y_pred.shape[0], -1)
            y_true = y_true.reshape(y_true.shape[0], -1)

        # Now, val_y_pred and val_y_true should be of shape (633, num_labels) or (633, num_tasks * num_labels)
        return pd.DataFrame({
            'pred': list(y_pred.flatten()),  # Flatten the predictions if needed
            'known': list(y_true.flatten())  # Flatten the true values if needed
        })
