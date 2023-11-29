import numpy as np
from evaluation.metrics.add_acc_metric import AddAccMetric, _precision_recall


def test_precision_recall_missing_classes():
    """Case where one class (here 0) is neither predicted nor labeled."""
    scores = [
        np.array([0.2, 0.2, 0.6]),  # 2
        np.array([0.3, 0.3, 0.4]),  # 2
        np.array([0.1, 0.8, 0.1]),  # 1
    ]
    labels = [1, 2, 1]
    precisions, recalls = _precision_recall(scores, labels)
    assert precisions.shape == (3,)
    assert recalls.shape == (3,)
    assert np.allclose(precisions, [1.0, 1.0, 0.5])
    assert np.allclose(recalls, [1.0, 0.5, 1.0])


def test_precision_recall_missing_class_predictions():
    scores = [
        np.array([0.2, 0.2, 0.6]),  # 2
        np.array([0.3, 0.3, 0.4]),  # 2
        np.array([0.1, 0.8, 0.1]),  # 1
    ]
    labels = [0, 1, 2]
    precisions, recalls = _precision_recall(scores, labels)
    assert precisions.shape == (3,)
    assert recalls.shape == (3,)
    assert np.allclose(precisions, [0.0, 0.0, 0.0])
    assert np.allclose(recalls, [0.0, 0.0, 0.0])


def test_f1_missing_classes():
    scores = [
        np.array([0.2, 0.2, 0.6]),  # 2
        np.array([0.3, 0.3, 0.4]),  # 2
        np.array([0.1, 0.8, 0.1]),  # 1
    ]
    labels = [1, 2, 1]
    metric = AddAccMetric(
        metric_list=("per_class_f1", "unweighted_average_f1", "weighted_average_f1")
    )
    eval_results = metric.calculate(scores, labels)
    assert eval_results["class_0_f1"] == 1.0
    assert eval_results["class_1_f1"] == 1.0 / 1.5
    assert eval_results["class_2_f1"] == 1.0 / 1.5
    assert eval_results["unweighted_average_f1"] == np.mean([1.0, 1.0 / 1.5, 1.0 / 1.5])
    assert eval_results["weighted_average_f1"] == np.average(
        [1.0, 1.0 / 1.5, 1.0 / 1.5], weights=[0, 2 / 3, 1 / 3]
    )


def test_f1_missing_class_predictions():
    scores = [
        np.array([0.2, 0.2, 0.6]),  # 2
        np.array([0.3, 0.3, 0.4]),  # 2
        np.array([0.1, 0.8, 0.1]),  # 1
    ]
    labels = [0, 1, 2]
    metric = AddAccMetric(
        metric_list=("per_class_f1", "unweighted_average_f1", "weighted_average_f1")
    )
    eval_results = metric.calculate(scores, labels)
    assert eval_results["class_0_f1"] == 0.0
    assert eval_results["class_1_f1"] == 0.0
    assert eval_results["class_2_f1"] == 0.0
    assert eval_results["unweighted_average_f1"] == 0.0
    assert eval_results["weighted_average_f1"] == 0.0
