# import numpy as np
# from numpy import array
# from sklearn.metrics import roc_auc_score, average_precision_score
# from typing import Dict

# import torch

# class Evaluator:
#     """Evaluator for the prediction performance."""

#     def __init__(self, name: str = "hiv", num_tasks: int = 1, eval_metric: str = "rocauc"):
#         """
#         Args:
#             name (str, optional): The name of the dataset. Defaults to "hiv".
#             num_tasks (int, optional): Number of tasks in the dataset. Defaults to 1.
#             eval_metric (str, optional): Metrics for the evaluation. Defaults to "rocauc".
#                 Metrics include : 'rocauc', 'ap', 'rmse', 'mae', 'acc', 'F1'.
#         """
#         self.name = name
#         self.num_tasks = num_tasks
#         self.eval_metric = eval_metric

#     def _parse_and_check_input(self, input_dict: Dict[np.ndarray, np.ndarray]):
#         """Evaluate the performance of the input_dict.

#         Args:
#             input_dict (Dict[np.ndarray, np.ndarray]): The true value and the predict
#                 value of the dataset. The format of input_dict is like:
#                 input_dict = {"y_true": y_true, "y_pred": y_pred}.

#         Returns:
#             y_true, y_pred: The true value and the predict value of the dataset.
#         """
#         if (
#             self.eval_metric == "rocauc"
#             or self.eval_metric == "ap"
#             or self.eval_metric == "rmse"
#             or self.eval_metric == "mae"
#             or self.eval_metric == "acc"
#         ):
#             if not "y_true" in input_dict:
#                 raise RuntimeError("Missing key of y_true")
#             if not "y_pred" in input_dict:
#                 raise RuntimeError("Missing key of y_pred")

#             y_true, y_pred = input_dict["y_true"], input_dict["y_pred"]

#             """
#                 y_true: numpy ndarray or torch tensor of shape (num_graph, num_tasks)
#                 y_pred: numpy ndarray or torch tensor of shape (num_graph, num_tasks)
#             """

#             # converting to torch.Tensor to numpy on cpu
#             if isinstance(y_true, torch.Tensor):
#                 y_true = y_true.detach().cpu().numpy()

#             if isinstance(y_pred, torch.Tensor):
#                 y_pred = y_pred.detach().cpu().numpy()

#             ## check type
#             if not isinstance(y_true, np.ndarray):
#                 raise RuntimeError(
#                     "Arguments to Evaluator need to be either numpy ndarray or torch tensor"
#                 )

#             if not y_true.shape == y_pred.shape:
#                 raise RuntimeError("Shape of y_true and y_pred must be the same")

#             if not y_true.ndim == 2:
#                 raise RuntimeError(
#                     "y_true and y_pred mush to 2-dim arrray, {}-dim array given".format(
#                         y_true.ndim
#                     )
#                 )

#             if not y_true.shape[1] == self.num_tasks:
#                 raise RuntimeError(
#                     "Number of tasks for {} should be {} but {} given".format(
#                         self.name, self.num_tasks, y_true.shape[1]
#                     )
#                 )

#             return y_true, y_pred

#         elif self.eval_metric == "F1":
#             if not "seq_ref" in input_dict:
#                 raise RuntimeError("Missing key of seq_ref")
#             if not "seq_pred" in input_dict:
#                 raise RuntimeError("Missing key of seq_pred")

#             seq_ref, seq_pred = input_dict["seq_ref"], input_dict["seq_pred"]

#             if not isinstance(seq_ref, list):
#                 raise RuntimeError("seq_ref must be of type list")

#             if not isinstance(seq_pred, list):
#                 raise RuntimeError("seq_pred must be of type list")

#             if len(seq_ref) != len(seq_pred):
#                 raise RuntimeError("Length of seq_true and seq_pred should be the same")

#             return seq_ref, seq_pred

#         else:
#             raise ValueError("Undefined eval metric %s " % (self.eval_metric))

#     def eval(self, input_dict: Dict[np.ndarray, np.ndarray]):
#         """Evaluate the performance of the input_dict.

#         Args:
#             input_dict (Dict[np.ndarray, np.ndarray]): The true value and the predict
#                 value of the dataset. The format of input_dict is like:
#                 input_dict = {"y_true": y_true, "y_pred": y_pred}

#         Returns:
#             A scalar value of the selected metric.
#         """
#         if self.eval_metric == "rocauc":
#             y_true, y_pred = self._parse_and_check_input(input_dict)
#             return self._eval_rocauc(y_true, y_pred)
#         if self.eval_metric == "ap":
#             y_true, y_pred = self._parse_and_check_input(input_dict)
#             return self._eval_ap(y_true, y_pred)
#         elif self.eval_metric == "rmse":
#             y_true, y_pred = self._parse_and_check_input(input_dict)
#             return self._eval_rmse(y_true, y_pred)
#         elif self.eval_metric == "mae":
#             y_true, y_pred = self._parse_and_check_input(input_dict)
#             return self._eval_mae(y_true, y_pred)
#         elif self.eval_metric == "acc":
#             y_true, y_pred = self._parse_and_check_input(input_dict)
#             return self._eval_acc(y_true, y_pred)
#         elif self.eval_metric == "F1":
#             seq_ref, seq_pred = self._parse_and_check_input(input_dict)
#             return self._eval_F1(seq_ref, seq_pred)
#         else:
#             raise ValueError("Undefined eval metric %s " % (self.eval_metric))

#     @property
#     def expected_input_format(self):
#         desc = "==== Expected input format of Evaluator for {}\n".format(self.name)
#         if self.eval_metric == "rocauc" or self.eval_metric == "ap":
#             desc += "{'y_true': y_true, 'y_pred': y_pred}\n"
#             desc += "- y_true: numpy ndarray or torch tensor of shape (num_graph, num_task)\n"
#             desc += "- y_pred: numpy ndarray or torch tensor of shape (num_graph, num_task)\n"
#             desc += "where y_pred stores score values (for computing AUC score),\n"
#             desc += "num_task is {}, and ".format(self.num_tasks)
#             desc += "each row corresponds to one graph.\n"
#             desc += "nan values in y_true are ignored during evaluation.\n"
#         elif self.eval_metric == "rmse":
#             desc += "{'y_true': y_true, 'y_pred': y_pred}\n"
#             desc += "- y_true: numpy ndarray or torch tensor of shape (num_graph, num_task)\n"
#             desc += "- y_pred: numpy ndarray or torch tensor of shape (num_graph, num_task)\n"
#             desc += "where num_task is {}, and ".format(self.num_tasks)
#             desc += "each row corresponds to one graph.\n"
#             desc += "nan values in y_true are ignored during evaluation.\n"
#         elif self.eval_metric == "acc":
#             desc += "{'y_true': y_true, 'y_pred': y_pred}\n"
#             desc += "- y_true: numpy ndarray or torch tensor of shape (num_node, num_task)\n"
#             desc += "- y_pred: numpy ndarray or torch tensor of shape (num_node, num_task)\n"
#             desc += "where y_pred stores predicted class label (integer),\n"
#             desc += "num_task is {}, and ".format(self.num_tasks)
#             desc += "each row corresponds to one graph.\n"
#         elif self.eval_metric == "F1":
#             desc += "{'seq_ref': seq_ref, 'seq_pred': seq_pred}\n"
#             desc += "- seq_ref: a list of lists of strings\n"
#             desc += "- seq_pred: a list of lists of strings\n"
#             desc += "where seq_ref stores the reference sequences of sub-tokens, and\n"
#             desc += "seq_pred stores the predicted sequences of sub-tokens.\n"
#         else:
#             raise ValueError("Undefined eval metric %s " % (self.eval_metric))

#         return desc

#     @property
#     def expected_output_format(self):
#         desc = "==== Expected output format of Evaluator for {}\n".format(self.name)
#         if self.eval_metric == "rocauc":
#             desc += "{'rocauc': rocauc}\n"
#             desc += "- rocauc (float): ROC-AUC score averaged across {} task(s)\n".format(
#                 self.num_tasks
#             )
#         elif self.eval_metric == "ap":
#             desc += "{'ap': ap}\n"
#             desc += (
#                 "- ap (float): Average Precision (AP) score averaged across {} task(s)\n".format(
#                     self.num_tasks
#                 )
#             )
#         elif self.eval_metric == "rmse":
#             desc += "{'rmse': rmse}\n"
#             desc += "- rmse (float): root mean squared error averaged across {} task(s)\n".format(
#                 self.num_tasks
#             )
#         elif self.eval_metric == "acc":
#             desc += "{'acc': acc}\n"
#             desc += "- acc (float): Accuracy score averaged across {} task(s)\n".format(
#                 self.num_tasks
#             )
#         elif self.eval_metric == "F1":
#             desc += "{'F1': F1}\n"
#             desc += "- F1 (float): F1 score averaged over samples.\n"
#         else:
#             raise ValueError("Undefined eval metric %s " % (self.eval_metric))

#         return desc

#     def _eval_rocauc(self, y_true: np.ndarray, y_pred: np.ndarray):
#         """compute ROC-AUC averaged across tasks.

#         Args:
#             y_true (np.ndarray): The true label of the dataset.
#             y_pred (np.ndarray): The predict label of the dataset.
#         """
#         rocauc_list = []

#         for i in range(y_true.shape[1]):
#             # AUC is only defined when there is at least one positive data.
#             if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
#                 # ignore nan values
#                 is_labeled = y_true[:, i] == y_true[:, i]
#                 rocauc_list.append(roc_auc_score(y_true[is_labeled, i], y_pred[is_labeled, i]))

#         if len(rocauc_list) == 0:
#             raise RuntimeError("No positively labeled data available. Cannot compute ROC-AUC.")

#         return {"rocauc": sum(rocauc_list) / len(rocauc_list)}

#     def _eval_ap(self, y_true: np.ndarray, y_pred: np.ndarray):
#         """compute Average Precision (AP) averaged across tasks.

#         Args:
#             y_true (np.ndarray): The true label of the dataset.
#             y_pred (np.ndarray): The predict label of the dataset.
#         """

#         ap_list = []

#         for i in range(y_true.shape[1]):
#             # AUC is only defined when there is at least one positive data.
#             if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
#                 # ignore nan values
#                 is_labeled = y_true[:, i] == y_true[:, i]
#                 ap = average_precision_score(y_true[is_labeled, i], y_pred[is_labeled, i])

#                 ap_list.append(ap)

#         if len(ap_list) == 0:
#             raise RuntimeError(
#                 "No positively labeled data available. Cannot compute Average Precision."
#             )

#         return {"ap": sum(ap_list) / len(ap_list)}

#     def _eval_rmse(self, y_true: np.ndarray, y_pred: np.ndarray):
#         """compute RMSE averaged across tasks.

#         Args:
#             y_true (np.ndarray): The true label of the dataset.
#             y_pred (np.ndarray): The predict label of the dataset.
#         """
#         rmse_list = []

#         for i in range(y_true.shape[1]):
#             # ignore nan values
#             is_labeled = y_true[:, i] == y_true[:, i]
#             rmse_list.append(np.sqrt(((y_true[is_labeled] - y_pred[is_labeled]) ** 2).mean()))

#         return {"rmse": sum(rmse_list) / len(rmse_list)}

#     def _eval_mae(self, y_true: np.ndarray, y_pred: np.ndarray):
#         """compute MAE averaged across tasks.

#         Args:
#             y_true (np.ndarray): The true label of the dataset.
#             y_pred (np.ndarray): The predict label of the dataset.
#         """
#         from sklearn.metrics import mean_absolute_error

#         mae_list = []
#         for i in range(y_true.shape[1]):
#             # ignore nan values
#             is_labeled = y_true[:, i] == y_true[:, i]
#             mae_list.append(mean_absolute_error(y_true[is_labeled], y_pred[is_labeled]))
#         return {"mae": sum(mae_list) / len(mae_list)}

#     def _eval_acc(self, y_true: array, y_pred: array):
#         """compute accuracy averaged across tasks.

#         Args:
#             y_true (np.ndarray): The true label of the dataset.
#             y_pred (np.ndarray): The predict label of the dataset.
#         """
#         acc_list = []

#         for i in range(y_true.shape[1]):
#             is_labeled = y_true[:, i] == y_true[:, i]
#             correct = y_true[is_labeled, i] == y_pred[is_labeled, i]
#             acc_list.append(float(np.sum(correct)) / len(correct))

#         return {"acc": sum(acc_list) / len(acc_list)}

#     def _eval_F1(self, seq_ref: np.ndarray, seq_pred: np.ndarray):
#         """compute F1 score averaged across tasks.

#         Args:
#             seq_ref (np.ndarray): The true label of the dataset.
#             seq_pred (np.ndarray): The predict label of the dataset.
#         """

#         precision_list = []
#         recall_list = []
#         f1_list = []

#         for l, p in zip(seq_ref, seq_pred):
#             label = set(l)
#             prediction = set(p)
#             true_positive = len(label.intersection(prediction))
#             false_positive = len(prediction - label)
#             false_negative = len(label - prediction)

#             if true_positive + false_positive > 0:
#                 precision = true_positive / (true_positive + false_positive)
#             else:
#                 precision = 0

#             if true_positive + false_negative > 0:
#                 recall = true_positive / (true_positive + false_negative)
#             else:
#                 recall = 0
#             if precision + recall > 0:
#                 f1 = 2 * precision * recall / (precision + recall)
#             else:
#                 f1 = 0

#             precision_list.append(precision)
#             recall_list.append(recall)
#             f1_list.append(f1)

#         return {
#             "precision": np.average(precision_list),
#             "recall": np.average(recall_list),
#             "F1": np.average(f1_list),
#         }


# import numpy as np
# from sklearn.metrics import (
#     roc_auc_score,
#     average_precision_score,
#     mean_squared_error,
#     mean_absolute_error,
#     accuracy_score,
#     f1_score,
#     r2_score,
# )
# from typing import Dict, Callable, Tuple

# class Evaluator:
#     """Evaluator for multi-task prediction performance in medicinal chemistry datasets."""

#     def __init__(self, name: str = "hiv", num_tasks: int = 1, eval_metric: str = "rocauc"):
#         """
#         Args:
#             name: Dataset name (e.g., 'hiv' for HIV bioactivity prediction).
#             num_tasks: Number of tasks in the dataset.
#             eval_metric: Evaluation metric ('rocauc', 'ap', 'rmse', 'mae', 'acc', 'f1').
#         """
#         self.name = name
#         self.num_tasks = num_tasks
#         self.eval_metric = eval_metric
#         self.metric_fns = {
#             "rocauc": self._eval_rocauc,
#             "ap": self._eval_ap,
#             "rmse": self._eval_rmse,
#             "mae": self._eval_mae,
#             "acc": self._eval_acc,
#             "f1": self._eval_f1,
#         }
#         if eval_metric not in self.metric_fns:
#             raise ValueError(f"Unsupported metric: {eval_metric}")

#     def _parse_and_check_input(self, input_dict: Dict) -> Tuple[np.ndarray, np.ndarray]:
#         """Validate and parse input dictionary."""
#         if "y_true" not in input_dict or "y_pred" not in input_dict:
#             raise KeyError("Input must contain 'y_true' and 'y_pred'")

#         y_true, y_pred = input_dict["y_true"], input_dict["y_pred"]

#         # Convert to numpy if torch tensor
#         if not isinstance(y_true, np.ndarray):
#             y_true = np.asarray(y_true)
#         if not isinstance(y_pred, np.ndarray):
#             y_pred = np.asarray(y_pred)

#         # Validate shapes
#         if y_true.shape != y_pred.shape:
#             raise ValueError(f"y_true shape {y_true.shape} != y_pred shape {y_pred.shape}")
#         if y_true.ndim != 2 or y_true.shape[1] != self.num_tasks:
#             raise ValueError(f"Expected shape (num_samples, {self.num_tasks}), got {y_true.shape}")

#         # Check for invalid values in y_pred
#         if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
#             raise ValueError("y_pred contains NaN or infinite values")

#         return y_true, y_pred

#     def _compute_task_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, metric_fn: Callable) -> float:
#         """Compute metric across tasks using masking, with weighted averaging."""
#         results, weights = [], []
#         for task_idx in range(self.num_tasks):
#             # Create mask for valid (non-NaN) labels
#             mask = ~np.isnan(y_true[:, task_idx])
#             if not mask.any():
#                 continue  # Skip tasks with no valid data

#             # Apply metric to valid data
#             try:
#                 score = metric_fn(y_true[mask, task_idx], y_pred[mask, task_idx])
#                 results.append(score)
#                 weights.append(mask.sum())
#             except ValueError as e:
#                 # Handle cases like insufficient labels for ROC-AUC
#                 continue

#         if not results:
#             raise RuntimeError(f"No valid data for {self.eval_metric} across {self.num_tasks} tasks")

#         # Weighted average by number of valid samples
#         return float(np.average(results, weights=weights))

#     def _eval_rocauc(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
#         """Compute ROC-AUC averaged across tasks."""
#         def rocauc_fn(true, pred):
#             if not (np.any(true == 1) and np.any(true == 0)):
#                 raise ValueError("Need both positive and negative labels")
#             return roc_auc_score(true, pred)
#         return {"rocauc": self._compute_task_metrics(y_true, y_pred, rocauc_fn)}

#     def _eval_ap(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
#         """Compute Average Precision averaged across tasks."""
#         return {"ap": self._compute_task_metrics(y_true, y_pred, average_precision_score)}

#     def _eval_rmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
#         """Compute RMSE averaged across tasks."""
#         return {"rmse": self._compute_task_metrics(y_true, y_pred, lambda t, p: np.sqrt(mean_squared_error(t, p)))}

#     def _eval_mae(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
#         """Compute MAE averaged across tasks."""
#         return {"mae": self._compute_task_metrics(y_true, y_pred, mean_absolute_error)}

#     def _eval_acc(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
#         """Compute accuracy averaged across tasks."""
#         return {"acc": self._compute_task_metrics(y_true, y_pred, accuracy_score)}

#     def _eval_f1(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
#         """Compute F1 score averaged across tasks."""
#         return {"f1": self._compute_task_metrics(y_true, y_pred, f1_score)}

#     def eval(self, input_dict: Dict) -> Dict[str, float]:
#         """Evaluate performance using the specified metric."""
#         y_true, y_pred = self._parse_and_check_input(input_dict)
#         return self.metric_fns[self.eval_metric](y_true, y_pred)

#     @property
#     def expected_input_format(self) -> str:
#         """Return expected input format description."""
#         desc = f"==== Expected input format for {self.name}\n"
#         desc += "{'y_true': y_true, 'y_pred': y_pred}\n"
#         desc += f"- y_true: numpy array of shape (num_samples, {self.num_tasks})\n"
#         desc += f"- y_pred: numpy array of shape (num_samples, {self.num_tasks})\n"
#         if self.eval_metric in ["rocauc", "ap", "f1"]:
#             desc += "- y_true: binary labels (0 or 1)\n"
#             desc += "- y_pred: probabilities for rocauc/ap, integer labels for f1\n"
#         elif self.eval_metric == "acc":
#             desc += "- y_true: integer class labels\n"
#             desc += "- y_pred: integer class labels\n"
#         elif self.eval_metric in ["rmse", "mae"]:
#             desc += "- y_true: continuous values\n"
#             desc += "- y_pred: continuous values\n"
#         return desc


# import numpy as np
# from sklearn.metrics import (
#     roc_auc_score, average_precision_score,
#     mean_squared_error, mean_absolute_error,
#     accuracy_score, f1_score, r2_score
# )
# from typing import Dict, Callable, Tuple

# class Evaluator:
#     """Evaluator for multi-task prediction performance."""
    
#     def __init__(self, name: str = "hiv", num_tasks: int = 1, eval_metric: str = "rocauc"):
#         self.name = name
#         self.num_tasks = num_tasks
#         self.eval_metric = eval_metric
#         self.metric_fns = {
#             "rocauc": self._eval_rocauc,
#             "ap": self._wrap(average_precision_score, "ap"),
#             "rmse": self._wrap(lambda t, p: np.sqrt(mean_squared_error(t, p)), "rmse"),
#             "mae": self._wrap(mean_absolute_error, "mae"),
#             "acc": self._wrap(accuracy_score, "acc"),
#             "f1": self._wrap(f1_score, "f1"),
#             "f1": self._wrap(r2_score, "f1"),
#         }
#         if eval_metric not in self.metric_fns:
#             raise ValueError(f"Unsupported metric: {eval_metric}")

#     def _parse_input(self, input_dict: Dict) -> Tuple[np.ndarray, np.ndarray]:
#         if "y_true" not in input_dict or "y_pred" not in input_dict:
#             raise KeyError("Input must contain 'y_true' and 'y_pred'")

#         y_true, y_pred = np.asarray(input_dict["y_true"]), np.asarray(input_dict["y_pred"])

#         if y_true.shape != y_pred.shape or y_true.ndim != 2 or y_true.shape[1] != self.num_tasks:
#             raise ValueError(f"Invalid shapes: got {y_true.shape}, expected (n_samples, {self.num_tasks})")

#         if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
#             raise ValueError("y_pred contains NaN or inf")

#         return y_true, y_pred

#     def _compute(self, y_true: np.ndarray, y_pred: np.ndarray, metric_fn: Callable) -> float:
#         scores, weights = [], []
#         for i in range(self.num_tasks):
#             mask = ~np.isnan(y_true[:, i])
#             if not mask.any():
#                 continue
#             try:
#                 scores.append(metric_fn(y_true[mask, i], y_pred[mask, i]))
#                 weights.append(mask.sum())
#             except ValueError:
#                 continue
#         if not scores:
#             raise RuntimeError(f"No valid data for {self.eval_metric}")
#         return float(np.average(scores, weights=weights))

#     def _wrap(self, fn: Callable, key: str) -> Callable:
#         return lambda y_true, y_pred: {key: self._compute(y_true, y_pred, fn)}

#     def _eval_rocauc(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
#         def safe_rocauc(t, p):
#             if not (np.any(t == 1) and np.any(t == 0)):
#                 raise ValueError
#             return roc_auc_score(t, p)
#         return {"rocauc": self._compute(y_true, y_pred, safe_rocauc)}

#     def eval(self, input_dict: Dict) -> Dict[str, float]:
#         y_true, y_pred = self._parse_input(input_dict)
#         return self.metric_fns[self.eval_metric](y_true, y_pred)

#     @property
#     def expected_input_format(self) -> str:
#         desc = f"==== Expected input format for {self.name}\n"
#         desc += "{'y_true': y_true, 'y_pred': y_pred}\n"
#         desc += f"- y_true, y_pred: shape (n_samples, {self.num_tasks})\n"
#         if self.eval_metric in ["rocauc", "ap", "f1"]:
#             desc += "- y_true: binary labels; y_pred: probabilities (rocauc/ap) or labels (f1)\n"
#         elif self.eval_metric == "acc":
#             desc += "- y_true, y_pred: integer labels\n"
#         else:
#             desc += "- y_true, y_pred: continuous values\n"
#         return desc



import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    r2_score,
)
from typing import Dict, Callable, Tuple

class Evaluator:
    """
    Evaluator for multi-task prediction performance in medicinal chemistry datasets.

    Supports classification (rocauc, ap, acc, f1, balanced_acc, mcc) and regression (rmse, mae)
    metrics for multi-task datasets. Handles missing labels via masking and weights task averages
    by the number of valid samples. Designed for bioactivity prediction tasks (e.g., HIV dataset).

    Args:
        name: Dataset name (e.g., 'hiv' for HIV bioactivity prediction).
        num_tasks: Number of tasks in the dataset (columns in y_true/y_pred).
        eval_metric: Evaluation metric ('rocauc', 'ap', 'rmse', 'mae', 'acc', 'f1', 'balanced_acc', 'mcc').

    Example:
        evaluator = Evaluator(name="hiv", num_tasks=2, eval_metric="rocauc")
        input_dict = {
            "y_true": np.array([[1, 0], [0, 1], [1, np.nan]]),
            "y_pred": np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3]])
        }
        result = evaluator.eval(input_dict)  # {'rocauc': 0.95}
    """

    def __init__(self, name: str = "hiv", num_tasks: int = 1, eval_metric: str = "rocauc"):
        self.name = name
        self.num_tasks = num_tasks
        self.eval_metric = eval_metric
        self.metric_fns = {
            "rocauc": self._eval_rocauc,
            "ap": self._wrap(average_precision_score, "ap"),
            "rmse": self._wrap(lambda t, p: np.sqrt(mean_squared_error(t, p)), "rmse"),
            "mae": self._wrap(mean_absolute_error, "mae"),
            "acc": self._wrap(accuracy_score, "acc"),
            "f1": self._wrap(lambda t, p: f1_score(t, p, average="macro"), "f1"),
            "balanced_acc": self._wrap(balanced_accuracy_score, "balanced_acc"),
            "mcc": self._wrap(matthews_corrcoef, "mcc"),
            "r2": self._wrap(r2_score, "r2"),
        }
        if eval_metric not in self.metric_fns:
            raise ValueError(f"Unsupported metric: {eval_metric}")

    def _parse_input(self, input_dict: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Validate and parse input dictionary.

        Args:
            input_dict: Dictionary with 'y_true' and 'y_pred' keys, each containing
                a numpy array or convertible (e.g., torch tensor) of shape (n_samples, num_tasks).

        Returns:
            Tuple of (y_true, y_pred) as numpy arrays.

        Raises:
            KeyError: If 'y_true' or 'y_pred' is missing.
            ValueError: If shapes are invalid, y_pred contains NaN/inf, or y_true has invalid labels.
        """
        if "y_true" not in input_dict or "y_pred" not in input_dict:
            raise KeyError("Input must contain 'y_true' and 'y_pred'")

        y_true, y_pred = np.asarray(input_dict["y_true"]), np.asarray(input_dict["y_pred"])

        if y_true.shape != y_pred.shape or y_true.ndim != 2 or y_true.shape[1] != self.num_tasks:
            raise ValueError(
                f"Invalid shapes: got {y_true.shape}, expected (n_samples, {self.num_tasks})"
            )

        if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
            raise ValueError("y_pred contains NaN or infinite values")

        # Validate labels for classification metrics
        if self.eval_metric in ["rocauc", "ap", "f1", "acc", "balanced_acc", "mcc"]:
            valid_labels = np.all(
                np.isin(y_true[~np.isnan(y_true)], np.unique(y_true[~np.isnan(y_true)]))
            )
            if not valid_labels:
                raise ValueError(
                    f"{self.eval_metric} requires integer labels in y_true (e.g., 0, 1 for binary)"
                )
            # Threshold y_pred for acc, f1, balanced_acc, mcc if probabilities are provided
            if self.eval_metric in ["acc", "f1", "balanced_acc", "mcc"]:
                if np.any((y_pred < 0) | (y_pred > 1)):
                    y_pred = (y_pred > 0.5).astype(int)

        return y_true, y_pred

    def _compute(self, y_true: np.ndarray, y_pred: np.ndarray, metric_fn: Callable) -> float:
        """
        Compute metric across tasks using masking, with weighted averaging.

        Args:
            y_true: Ground truth labels, shape (n_samples, num_tasks).
            y_pred: Predicted values, shape (n_samples, num_tasks).
            metric_fn: Function to compute the metric for a single task (e.g., roc_auc_score).

        Returns:
            Weighted average of metric scores across valid tasks.

        Raises:
            RuntimeError: If no tasks have valid data.
        """
        scores, weights, skipped_tasks = [], [], []
        for i in range(self.num_tasks):
            mask = ~np.isnan(y_true[:, i])
            if not mask.any():
                skipped_tasks.append(i)
                continue
            try:
                scores.append(metric_fn(y_true[mask, i], y_pred[mask, i]))
                weights.append(mask.sum())
            except ValueError as e:
                skipped_tasks.append(i)
                continue
        if not scores:
            raise RuntimeError(
                f"No valid data for {self.eval_metric}. Skipped tasks: {skipped_tasks}"
            )
        return float(np.average(scores, weights=weights))

    def _wrap(self, fn: Callable, key: str) -> Callable:
        """
        Wrap a metric function to compute it across tasks and return a dictionary.

        Args:
            fn: Metric function (e.g., roc_auc_score).
            key: Metric name for the output dictionary.

        Returns:
            Callable that computes the metric and returns {key: score}.
        """
        return lambda y_true, y_pred: {key: self._compute(y_true, y_pred, fn)}

    def _eval_rocauc(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Compute ROC-AUC averaged across tasks, supporting binary and multi-class classification.

        Args:
            y_true: Ground truth binary or integer labels, shape (n_samples, num_tasks).
            y_pred: Predicted probabilities, shape (n_samples, num_tasks).

        Returns:
            Dictionary with ROC-AUC score: {'rocauc': score}.
        """
        def safe_rocauc(t, p):
            if len(np.unique(t)) < 2:
                raise ValueError("Need at least two distinct labels")
            return (
                roc_auc_score(t, p, multi_class="ovr")
                if len(np.unique(t)) > 2
                else roc_auc_score(t, p)
            )
        return {"rocauc": self._compute(y_true, y_pred, safe_rocauc)}

    def eval(self, input_dict: Dict) -> Dict[str, float]:
        """
        Evaluate performance using the specified metric.

        Args:
            input_dict: Dictionary with 'y_true' and 'y_pred' keys.

        Returns:
            Dictionary with the metric score (e.g., {'rocauc': 0.95}).
        """
        y_true, y_pred = self._parse_input(input_dict)
        return self.metric_fns[self.eval_metric](y_true, y_pred)

    @property
    def expected_input_format(self) -> str:
        """
        Return expected input format description.

        Returns:
            String describing the expected input format for the evaluator.
        """
        desc = f"==== Expected input format for {self.name}\n"
        desc += "{'y_true': y_true, 'y_pred': y_pred}\n"
        desc += f"- y_true, y_pred: numpy array or convertible, shape (n_samples, {self.num_tasks})\n"
        desc += "- y_true: may contain NaN for missing labels\n"
        if self.eval_metric in ["rocauc", "ap"]:
            desc += "- y_true: binary (0, 1) or integer labels\n"
            desc += "- y_pred: probabilities (0 to 1)\n"
        elif self.eval_metric in ["f1", "acc", "balanced_acc", "mcc"]:
            desc += "- y_true: integer labels (e.g., 0, 1 for binary)\n"
            desc += "- y_pred: integer labels or probabilities (auto-thresholded at 0.5)\n"
        else:  # rmse, mae
            desc += "- y_true, y_pred: continuous values\n"
        return desc

    @property
    def expected_output_format(self) -> str:
        """
        Return expected output format description.

        Returns:
            String describing the expected output format for the evaluator.
        """
        desc = f"==== Expected output format for {self.name}\n"
        desc += f"{{'{self.eval_metric}': score}}\n"
        desc += f"- score (float): Weighted average {self.eval_metric} across {self.num_tasks} tasks\n"
        return desc