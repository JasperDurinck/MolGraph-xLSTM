import argparse
import math
import sys

import shlex

def set_args_from_string(arg_string):
    sys.argv = [sys.argv[0]] + shlex.split(arg_string)

def parse_args_dict(str_input='') -> dict:
    """Parses command-line arguments into a dictionary."""

    set_args_from_string(str_input)

    parser = argparse.ArgumentParser("Arguments for training")

    parser.add_argument("--gpu_index", type=str, default='3', help="CUDA_VISIBLE_DEVICES value")

    # Main task settings
    parser.add_argument("--classification", action="store_true", help="classification task")
    parser.add_argument("--global_feature", action="store_true", help="include global features")
    parser.add_argument("--batch_size", type=int, default=256, help="batch size")
    parser.add_argument("--epochs", type=int, default=1000, help="number of training epochs")

    # Optimization parameters
    parser.add_argument("--learning_rate", type=float, default=0.05, help="learning rate")
    parser.add_argument("--lr_decay_epochs", type=str, default="1000", help="comma-separated list of epochs for learning rate decay")
    parser.add_argument("--lr_decay_rate", type=float, default=0.1, help="decay rate for learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight decay for optimizer")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")

    # Dataset and I/O
    parser.add_argument("--data_file", type=str, default="freesolv.pth", help="dataset name")
    parser.add_argument("--processed_data_path", type=str, default='procesed_data.pth', help="path to store processed data")
    parser.add_argument("--dataname", type=str, default='', help="name of data set")
    parser.add_argument("--target_cols", type=str, default=["target"], help="target columns in dataframe")
    parser.add_argument("--smiles_col", type=str, default="smiles", help="smiles column in dataframe")
    parser.add_argument("--num_tasks", type=int, default=1, help="number of tasks for multitask learning")
    parser.add_argument("--recalc_features", type=bool, default=False, help="recalculate features if already in loaded data")

    parser.add_argument("--fit_train_subset", type=list, default=['train_idx'], help="list of key with indexes in data dict")
    parser.add_argument("--fit_val_subset", type=list, default=['val_idx'], help="list of key with indexes in data dict")
    parser.add_argument("--fit_test_subset", type=list, default=['test_idx'], help="list of key with indexes in data dict")
    
    parser.add_argument("--max_len_tokenizer", type=int, default=100, help="RobertaTokenizerFast ChemBERTa_zinc250k_v2_40k max_len")

    # Model architecture
    parser.add_argument("--mlp_layers", type=int, default=2, help="number of MLP layers")
    parser.add_argument("--num_gc_layers", type=int, default=3, help="number of GCN layers")
    parser.add_argument("--power", type=int, default=4, help="power parameter (e.g. jump knowledge layers)")
    parser.add_argument("--num_dim", type=int, default=64, help="embedding dimension")

    # Scheduler and warm-up
    parser.add_argument("--cosine", action="store_true", help="use cosine annealing scheduler")
    parser.add_argument("--warm", action="store_true", help="use learning rate warm-up")
    parser.add_argument("--trial", type=str, default="0", help="experiment ID for multiple runs")

    # XLSTM model configuration
    parser.add_argument("--num_blocks", type=int, default=2, help="number of XLSTM blocks")
    parser.add_argument("--slstm", nargs='+', type=int, default=[0], help="positions of slstm layers")

    # Classifier
    parser.add_argument("--dropout", type=float, default=0.5, help="dropout rate in classifier")
    parser.add_argument("--mlp_layer", type=int, default=2, help="number of layers in classifier MLP")
    parser.add_argument("--num_experts", type=int, default=8, help="number of MoE experts")
    parser.add_argument("--num_heads", type=int, default=8, help="number of MoE heads")

    # Saving
    parser.add_argument("--checkpoint_path", type=str, default='model_checkpoint.pth', help="model weights checkpoint")
    parser.add_argument("--z_scale_setting_save_path", type=str, default='scaling_params.pth', help="path to save z-scale settings")
    parser.add_argument("--re_calc_z_scale_settings", type=str2bool, default=False, help="recalulate z-scales settings")
    parser.add_argument("--track_best_model", type=str2bool, default=True, help="saves model weights of the best val metric score epoch during fitting")

    parser.add_argument("--score_metric", type=str, default='r2', help="score metric")

    args = vars(parser.parse_args())  # ← Convert Namespace to dict

    # Parse comma-separated decay epochs into a list of integers
    args['lr_decay_epochs'] = [int(e) for e in args['lr_decay_epochs'].split(",")]

    # Construct a base name for the model based on configuration
    base_name = f"lr_{args['learning_rate']}_bsz_{args['batch_size']}_trial_{args['trial']}_blocks_{args['num_blocks']}_slstm_{args['slstm']}_power_{args['power']}_dims_{args['num_dim']}"

    args["model_name"] = base_name
    if args["cosine"]:
        args["model_name"] += "_cosine"

    # Auto-enable warmup if using a very large batch
    if args["batch_size"] > 1024:
        args["warm"] = True

    if args["warm"]:
        args["model_name"] += "_warm"
        args["warmup_from"] = 0.01
        args["warm_epochs"] = 100

        if args["cosine"]:
            eta_min = args["learning_rate"] * (args["lr_decay_rate"] ** 3)
            args["warmup_to"] = eta_min + (args["learning_rate"] - eta_min) * (
                1 + math.cos(math.pi * args["warm_epochs"] / args["epochs"])
            ) / 2
        else:
            # Fall back gracefully if this isn't defined elsewhere
            args["warmup_to"] = args.get("learning_rate_gcn", args["learning_rate"])

    return args


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")