from collections import OrderedDict
from typing import Dict, List, Optional, Union

import numpy as np
from mmaction.evaluation import AccMetric
from mmaction.evaluation.functional.accuracy import confusion_matrix
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
        num_classes = preds[0].shape[1]

        for metric in self.metrics:
            if metric == "per_class_f1":
                for i in range(num_classes):
                    eval_results[f"class_{i}_f1"] = (2 * precisions[i] * recalls[i]) / (
                        precisions[i] + recalls[i]
                    )

            elif metric == "per_class_precision":
                for i in range(num_classes):
                    eval_results[f"class_{i}_precision"] = precisions[i]

            elif metric == "per_class_recall":
                for i in range(num_classes):
                    eval_results[f"class_{i}_recall"] = recalls[i]

            elif metric == "unweighted_average_f1":
                f1_scores = (2 * precisions * recalls) / (precisions + recalls)
                eval_results["unweighted_average_f1"] = np.nanmean(f1_scores)

            elif metric == "weighted_average_f1":
                f1_scores = (2 * precisions * recalls) / (precisions + recalls)
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
    num_classes = scores[0].shape[1]
    precisions = np.empty(num_classes)
    precisions.fill(np.nan)
    recalls = np.empty(num_classes)
    recalls.fill(np.nan)

    pred = np.argmax(scores, axis=1)
    # cf_matrix has predictions as columns and ground truth as rows
    cf_matrix = confusion_matrix(pred, labels).astype(float)
    # cf_matrix returns a smaller matrix if not all classes are present
    for i in range(cf_matrix.shape[0]):
        tp = cf_matrix[i, i]
        fp = cf_matrix[i, :].sum() - tp
        fn = cf_matrix[:, i].sum() - tp
        precision = tp / (tp + fp)
        precisions[i] = precision
        recall = tp / (tp + fn)
        recalls[i] = recall

    return precisions, recalls
