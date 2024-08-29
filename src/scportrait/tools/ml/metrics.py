import torch

def precision(predictions, labels, pos_label=0):
    """
    Calculate precision for predicting class `pos_label`.
 
    Parameters
    ----------
    predictions : torch.Tensor
        Model predictions.
    labels : torch.Tensor
        Ground truth labels.
    pos_label : int, optional, default = 0
        The positive label for which to calculate precision.

    Returns
    -------
    precision : float
        Precision for predicting class `pos_label`.
    """
    
    _, max_indices = torch.max(predictions,1)
    
    # bool array for all elements which equal the class label
    # all predicted positives
    all_pred_positives = max_indices == pos_label
    
    # masked ground truth tensor for all values predicetd to be positive
    masked = labels.masked_select(all_pred_positives)
    
    # true positives divided by all positives
    correct = (masked == pos_label).sum()
    precision = correct/len(masked)
    
    return precision

def recall(predictions, labels, pos_label=0):
    """
    Calculate recall for predicting class `pos_label`.
 
    Parameters
    ----------
    predictions : torch.Tensor
        Model predictions.
    labels : torch.Tensor
        Ground truth labels.
    pos_label : int, optional, default = 0
        The positive label for which to calculate precision.

    Returns
    -------
    recall : float
        Recall for predicting class `pos_label`.
    """
    
    _, max_indices = torch.max(predictions,1)
    
    # all true positive elements
    all_labled_positives = labels == pos_label
    
    # masked prediction tensor for all values ground truth positive
    masked = max_indices.masked_select(all_labled_positives)
    
    correct = (masked == pos_label).sum()
    recall = correct/len(masked)
    
    return recall