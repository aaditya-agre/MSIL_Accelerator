from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def classification_metrics(y_true, y_pred, average='binary'):
    """
    Computes common evaluation metrics for classification models.

    Parameters:
        y_true (array-like): True class labels.
        y_pred (array-like): Predicted class labels.
        average (str): Averaging method for multi-class/multi-label data.
                       Options: 'binary', 'macro', 'micro', 'weighted' (default is 'binary').

    Returns:
        dict: Dictionary containing Accuracy, Precision, Recall, and F1 Score.
    """
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'Recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'F1 Score': f1_score(y_true, y_pred, average=average, zero_division=0)
    }
