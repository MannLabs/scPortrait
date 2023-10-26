import torch


def precision(predictions, labels, pos_label=0):
    """
    Calculate precision for predicting class `pos_label`.

    Args:
        predictions (torch.Tensor): Model predictions.
        labels (torch.Tensor): Ground truth labels.
        pos_label (int, optional): The positive label for which to calculate precision. Defaults to 0.

    Returns:
        float: Precision for predicting class `pos_label`.
    """
    _, max_indices = torch.max(predictions, 1)

    # bool array for all elements which equal the class label
    # all predicted positives
    all_pred_positives = max_indices == pos_label

    # masked ground truth tensor for all values predicted to be positive
    masked = labels.masked_select(all_pred_positives)

    # true positives divided by all positives
    correct = (masked == pos_label).sum()
    precision = correct / len(masked)

    return precision


def recall(predictions, labels, pos_label=0):
    """
    Calculate recall for predicting class `pos_label`.

    Args:
        predictions (torch.Tensor): Model predictions.
        labels (torch.Tensor): Ground truth labels.
        pos_label (int, optional): The positive label for which to calculate precision. Defaults to 0.

    Returns:
        float: Recall for predicting class `pos_label`.
    """
    _, max_indices = torch.max(predictions, 1)

    # all true positive elements
    all_labeled_positives = labels == pos_label

    # masked prediction tensor for all values with ground truth positive
    masked = max_indices.masked_select(all_labeled_positives)

    correct = (masked == pos_label).sum()
    recall = correct / len(masked)

    return recall
