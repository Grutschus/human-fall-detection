from collections import OrderedDict
from typing import Dict, List, Optional, Union

import numpy as np
from mmaction.evaluation import AccMetric

from mmaction.registry import METRICS


@METRICS.register_module()
class AddAccMetric(AccMetric):
    """Additional accuracy metrics.

    Possible metrics are:
        - per_class_f1
        - per_class_precision
        - per_class_recall
        - unweighted_average_f1
        - weighhted_average_f1"""

    allowed_metrics = (
        "per_class_f1",
        "per_class_precision",
        "per_class_recall",
        "unweighted_average_f1",
        "weighted_average_f1",
    )

    def __init__(
        self,
        metric_list: Optional[Union[str, tuple[str, ...]]] = ("per_class_f1",),
        collect_device: str = "cpu",
        metric_options: Optional[Dict] = None,
        prefix: Optional[str] = None,
    ) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)

        if not isinstance(metric_list, (str, tuple)):
            raise TypeError(
                "metric_list must be str or tuple of str, "
                f"but got {type(metric_list)}"
            )

        if isinstance(metric_list, str):
            metrics = (metric_list,)
        else:
            metrics = metric_list  # type: ignore[assignment]

        for metric in metrics:
            if metric not in self.allowed_metrics:
                raise ValueError(
                    f"metric {metric} is not supported, "
                    f"supported metrics are {self.allowed_metrics}"
                )

        self.metrics = metrics
        self.metric_options = metric_options or {}

    def calculate(self, preds: List[np.ndarray], labels: List[int]) -> Dict:
        """Compute the metrics from processed results.

        Args:
            preds (list[np.ndarray]): List of the prediction scores.
            labels (list[int]): List of the labels.

        Returns:
            dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        eval_results = OrderedDict()

        # We need to calculate per class precision and recall for all metrics
        precisions, recalls = _precision_recall(preds, labels)
        f1_scores = np.divide(
            2 * precisions * recalls,
            precisions + recalls,
            out=np.zeros_like(precisions),
            where=(precisions + recalls) != 0,
        )
        num_classes = preds[0].shape[0]

        for metric in self.metrics:
            if metric == "per_class_f1":
                for i in range(num_classes):
                    eval_results[f"class_{i}_f1"] = f1_scores[i]

            elif metric == "per_class_precision":
                for i in range(num_classes):
                    eval_results[f"class_{i}_precision"] = precisions[i]

            elif metric == "per_class_recall":
                for i in range(num_classes):
                    eval_results[f"class_{i}_recall"] = recalls[i]

            elif metric == "unweighted_average_f1":
                eval_results["unweighted_average_f1"] = np.nanmean(f1_scores)

            elif metric == "weighted_average_f1":
                occurences = np.bincount(np.array(labels), minlength=num_classes)
                weights = occurences / len(labels)
                eval_results["weighted_average_f1"] = np.average(
                    f1_scores, weights=weights
                )

        return eval_results


def _precision_recall(
    scores: list[np.ndarray], labels: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate precision and recall per class.

    Args:
        scores (list[np.ndarray]): Prediction scores for each class.
        labels (list[int]): Ground truth labels.

    Returns:
        np.ndarray: Array of precisions.
        np.ndarray: Array of recalls.
    """
    # num_class is list[np.ndarray] where each element is a 1D array of scores
    num_classes = scores[0].shape[0]
    precisions = np.empty(num_classes)
    precisions.fill(np.nan)
    recalls = np.empty(num_classes)
    recalls.fill(np.nan)

    pred = np.argmax(scores, axis=1)
    # cf_matrix has predictions as columns and ground truth as rows
    cf_matrix = _confusion_matrix(pred, labels, num_classes=num_classes).astype(float)
    # cf_matrix returns a smaller matrix if not all classes are present
    for i in range(num_classes):
        tp = cf_matrix[i, i]
        fp = cf_matrix[:, i].sum() - tp
        fn = cf_matrix[i, :].sum() - tp
        # Edge cases:
        # https://github.com/dice-group/gerbil/wiki/Precision,-Recall-and-F1-measure#dividing-by-0
        if tp + fp + fn == 0:
            precisions[i] = 1.0
            recalls[i] = 1.0
            continue
        if tp + fp == 0:
            precisions[i] = 0.0
        else:
            precisions[i] = tp / (tp + fp)
        if tp + fn == 0:
            recalls[i] = 0.0
        else:
            recalls[i] = tp / (tp + fn)

    return precisions, recalls


def _confusion_matrix(y_pred, y_real, normalize=None, num_classes=None):
    """Compute confusion matrix.

    Args:
        y_pred (list[int] | np.ndarray[int]): Prediction labels.
        y_real (list[int] | np.ndarray[int]): Ground truth labels.
        normalize (str | None): Normalizes confusion matrix over the true
            (rows), predicted (columns) conditions or all the population.
            If None, confusion matrix will not be normalized. Options are
            "true", "pred", "all", None. Default: None.

    Returns:
        np.ndarray: Confusion matrix.
    """
    if normalize not in ["true", "pred", "all", None]:
        raise ValueError("normalize must be one of {'true', 'pred', " "'all', None}")

    if isinstance(y_pred, list):
        y_pred = np.array(y_pred)
        if y_pred.dtype == np.int32:
            y_pred = y_pred.astype(np.int64)
    if not isinstance(y_pred, np.ndarray):
        raise TypeError(f"y_pred must be list or np.ndarray, but got {type(y_pred)}")
    if not y_pred.dtype == np.int64:
        raise TypeError(f"y_pred dtype must be np.int64, but got {y_pred.dtype}")

    if isinstance(y_real, list):
        y_real = np.array(y_real)
        if y_real.dtype == np.int32:
            y_real = y_real.astype(np.int64)
    if not isinstance(y_real, np.ndarray):
        raise TypeError(f"y_real must be list or np.ndarray, but got {type(y_real)}")
    if not y_real.dtype == np.int64:
        raise TypeError(f"y_real dtype must be np.int64, but got {y_real.dtype}")

    if num_classes:
        label_set = np.arange(num_classes)
    else:
        label_set = np.unique(np.concatenate((y_pred, y_real)))

    num_labels = len(label_set)
    max_label = label_set[-1]
    label_map = np.zeros(max_label + 1, dtype=np.int64)
    for i, label in enumerate(label_set):
        label_map[label] = i

    y_pred_mapped = label_map[y_pred]
    y_real_mapped = label_map[y_real]

    confusion_mat = np.bincount(
        num_labels * y_real_mapped + y_pred_mapped, minlength=num_labels**2
    ).reshape(num_labels, num_labels)

    with np.errstate(all="ignore"):
        if normalize == "true":
            confusion_mat = confusion_mat / confusion_mat.sum(axis=1, keepdims=True)
        elif normalize == "pred":
            confusion_mat = confusion_mat / confusion_mat.sum(axis=0, keepdims=True)
        elif normalize == "all":
            confusion_mat = confusion_mat / confusion_mat.sum()
        confusion_mat = np.nan_to_num(confusion_mat)

    return confusion_mat
