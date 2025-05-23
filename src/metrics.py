import torch
from torch import Tensor
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score, mean_absolute_percentage_error
import numpy as np

def mse(y_true, y_pred):
    return F.mse_loss(y_true, y_pred, reduction='mean')

def rmse(y_true, y_pred):
    return torch.sqrt(F.mse_loss(y_true, y_pred, reduction='mean'))

def mae(y_true, y_pred):
    return F.l1_loss(y_true, y_pred, reduction='mean')

def r2_score(y_true, y_pred):
    return r2_score(y_true, y_pred)

def mape(y_true, y_pred):
    return mean_absolute_percentage_error(y_true, y_pred)

def accuracy(y_true, y_pred):
    y_pred = (y_pred > 0.5).int()  # Use .int() instead of .astype(int)
    return (y_true == y_pred).float().mean()

def precision(y_true, y_pred):
    y_pred = (torch.sigmoid(y_pred) > 0.3).int()  # Lower threshold
    yt = y_true.cpu().numpy()
    yp = y_pred.cpu().numpy()
    n_classes = yt.shape[1] if yt.ndim == 2 else int(max(yt.max(), yp.max())) + 1
    yt = to_multilabel_indicator(yt, n_classes)
    yp = to_multilabel_indicator(yp, n_classes)
    score = precision_score(
        yt, yp,
        average='samples', zero_division=0
    )
    return torch.tensor(score)

def recall(y_true, y_pred):
    y_pred = (torch.sigmoid(y_pred) > 0.3).int() 
    yt = y_true.cpu().numpy()
    yp = y_pred.cpu().numpy()
    n_classes = yt.shape[1] if yt.ndim == 2 else int(max(yt.max(), yp.max())) + 1
    yt = to_multilabel_indicator(yt, n_classes)
    yp = to_multilabel_indicator(yp, n_classes)
    score = recall_score(
        yt, yp,
        average='samples', zero_division=0
    )
    return torch.tensor(score)

def f1(y_true, y_pred):
    y_pred = (torch.sigmoid(y_pred) > 0.3).int() 
    yt = y_true.cpu().numpy()
    yp = y_pred.cpu().numpy()
    n_classes = yt.shape[1] if yt.ndim == 2 else int(max(yt.max(), yp.max())) + 1
    yt = to_multilabel_indicator(yt, n_classes)
    yp = to_multilabel_indicator(yp, n_classes)
    score = f1_score(
        yt, yp,
        average='samples', zero_division=0
    )
    return torch.tensor(score)

def to_multilabel_indicator(arr, n_classes=None):
    arr = np.asarray(arr)
    if arr.ndim == 1 or (arr.ndim == 2 and arr.shape[1] == 1):
        # Convert class indices to one-hot
        if n_classes is None:
            n_classes = int(arr.max()) + 1
        arr = np.eye(n_classes)[arr.reshape(-1)]
    # Ensure strictly 0/1 and int type
    arr = (arr > 0.5).astype(int)
    return arr