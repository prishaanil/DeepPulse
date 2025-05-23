import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import torch
from torch import nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from torch.utils.data import WeightedRandomSampler, DataLoader
from imblearn.over_sampling import SMOTE

from src.models.patchTST import PatchTST
from src.learner import Learner, transfer_weights
from src.callback.core import *
from src.callback.tracking import *
from src.callback.patch_mask import *
from src.callback.transforms import *
from src.metrics import *
from src.basics import set_device
from datautils import *

import argparse
print("Script started")

parser = argparse.ArgumentParser()
# Pretraining and Finetuning
parser.add_argument('--is_finetune', type=int, default=0, help='do finetuning or not')
parser.add_argument('--is_linear_probe', type=int, default=0, help='if linear_probe: only finetune the last layer')
# Dataset and dataloader
parser.add_argument('--dset_finetune', type=str, default='etth1', help='dataset name')
parser.add_argument('--context_points', type=int, default=512, help='sequence length')
parser.add_argument('--target_points', type=int, default=96, help='forecast horizon')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--num_workers', type=int, default=0, help='number of workers for DataLoader')
parser.add_argument('--scaler', type=str, default='standard', help='scale the input data')
parser.add_argument('--features', type=str, default='M', help='for multivariate model or univariate model')
# Patch
parser.add_argument('--patch_len', type=int, default=12, help='patch length')
parser.add_argument('--stride', type=int, default=12, help='stride between patch')
# RevIN
parser.add_argument('--revin', type=int, default=1, help='reversible instance normalization')
# Model args
parser.add_argument('--n_layers', type=int, default=3, help='number of Transformer layers')
parser.add_argument('--n_heads', type=int, default=16, help='number of Transformer heads')
parser.add_argument('--d_model', type=int, default=128, help='Transformer d_model')
parser.add_argument('--d_ff', type=int, default=256, help='Tranformer MLP dimension')
parser.add_argument('--dropout', type=float, default=0.2, help='Transformer dropout')
parser.add_argument('--head_dropout', type=float, default=0.2, help='head dropout')
# Optimization args
parser.add_argument('--n_epochs_finetune', type=int, default=20, help='number of finetuning epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
# Pretrained model name
parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained model name')
# model id to keep track of the number of models saved
parser.add_argument('--finetuned_model_id', type=int, default=1, help='id of the saved finetuned model')
parser.add_argument('--model_type', type=str, default='based_model', help='for multivariate model or univariate model')


args = parser.parse_args()
print('args:', args)
args.save_path = 'saved_models/' + args.dset_finetune + '/masked_patchtst/' + args.model_type + '/'
if not os.path.exists(args.save_path): os.makedirs(args.save_path)

suffix_name = '_cw'+str(args.context_points)+'_tw'+str(args.target_points) + '_patch'+str(args.patch_len) + '_stride'+str(args.stride) + '_epochs-finetune' + str(args.n_epochs_finetune) + '_model' + str(args.finetuned_model_id)
if args.is_finetune: args.save_finetuned_model = args.dset_finetune+'_patchtst_finetuned'+suffix_name
elif args.is_linear_probe: args.save_finetuned_model = args.dset_finetune+'_patchtst_linear-probe'+suffix_name
else: args.save_finetuned_model = args.dset_finetune+'_patchtst_finetuned'+suffix_name

# get available GPU devide
print("Before set_device()")
set_device()
print("Device set")

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # logits: (batch, num_classes), targets: (batch, num_classes)
        bce_loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class ClassBalancedLoss(nn.Module):
    def __init__(self, samples_per_cls, beta=0.9999, reduction='mean'):
        super().__init__()
        effective_num = 1.0 - np.power(beta, samples_per_cls)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * len(samples_per_cls)  # normalize
        self.weights = torch.tensor(weights, dtype=torch.float32)
        self.reduction = reduction

    def forward(self, logits, targets):
        weights = self.weights.to(logits.device)
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        # weights: (num_classes,) -> (batch, num_classes)
        loss = bce * weights
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class ClassBalancedFocalLoss(nn.Module):
    def __init__(self, samples_per_cls, beta=0.9999, gamma=2.0, reduction='mean'):
        super().__init__()
        effective_num = 1.0 - np.power(beta, samples_per_cls)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * len(samples_per_cls)  # normalize
        self.weights = torch.tensor(weights, dtype=torch.float32)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        weights = self.weights.to(logits.device)
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce)
        focal_factor = (1 - pt) ** self.gamma
        loss = weights * focal_factor * bce
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

def get_model(c_in, args, head_type, num_classes, weight_path=None):
    print("Creating model...")
    num_patch = (max(args.context_points, args.patch_len)-args.patch_len) // args.stride + 1    
    print('number of patches:', num_patch)
    model = PatchTST(
        c_in=c_in,
        target_dim=num_classes,
        patch_len=args.patch_len,
        stride=args.stride,
        num_patch=num_patch,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_model=args.d_model,
        shared_embedding=True,
        d_ff=args.d_ff,                        
        dropout=args.dropout,
        head_dropout=args.head_dropout,
        act='relu',
        head_type=head_type,
        res_attention=False
    )
    if weight_path: 
        print("Transferring weights...")
        model = transfer_weights(weight_path, model)
    print('number of model params', sum(p.numel() for p in model.parameters() if p.requires_grad))
    print("Model created")
    return model

def find_best_thresholds(learn, dls):
    print("Finding best thresholds...")
    y_true = []
    y_prob = []
    for xb, yb in dls.valid:
        with torch.no_grad():
            logits = learn.model(xb.to('cuda'))
            probs = torch.sigmoid(logits).cpu().numpy()
        y_true.append(yb.numpy())
        y_prob.append(probs)
    y_true = np.concatenate(y_true)
    y_prob = np.concatenate(y_prob)
    n_classes = y_true.shape[1]
    best_thrs = np.full(n_classes, 0.5)
    print("Validation set support per class:", y_true.sum(axis=0))
    for i in range(n_classes):
        best_f1 = 0
        for thr in np.arange(0.01, 0.5, 0.01):
            y_pred = (y_prob[:, i] > thr).astype(int)
            f1 = f1_score(y_true[:, i], y_pred, zero_division=0)
            print(f"Class {i}, Threshold {thr:.2f}, F1: {f1:.4f}")
            if f1 > best_f1:
                best_f1 = f1
                best_thrs[i] = thr
    print("Best thresholds per class:", best_thrs)
    print("Best threshold search complete")
    return best_thrs

def find_lr(head_type):
    print("Finding learning rate...")
    print("Getting dataloaders...")
    dls, num_channels, seq_len = get_dls(args)  # <-- Unpack all returned values
    print("Dataloaders ready")
    model = get_model(num_channels, args, head_type, num_classes=len(dls.train.dataset.all_codes))
    print("Transferring weights for LR finder...")
    model = transfer_weights(args.pretrained_model, model)
    loss_func = FocalLoss(alpha=1, gamma=2)
    cbs = [RevInCB(num_channels, denorm=True)] if args.revin else []
    cbs += [PatchCB(patch_len=args.patch_len, stride=args.stride)]
    learn = Learner(dls, model, loss_func, lr=args.lr, cbs=cbs)
    print("Starting LR finder...")
    suggested_lr = learn.lr_finder()
    print('suggested_lr', suggested_lr)
    return suggested_lr

def save_recorders(learn):
    print("Saving recorder logs...")
    train_loss = learn.recorder['train_loss']
    valid_loss = learn.recorder['valid_loss']
    df = pd.DataFrame(data={'train_loss': train_loss, 'valid_loss': valid_loss})
    df.to_csv(args.save_path + args.save_finetuned_model + '_losses.csv', float_format='%.6f', index=False)
    print("Recorder logs saved.")

def get_dls_with_oversampling(args, oversample_power=5.0):
    print("Starting get_dls_with_oversampling...")
    dls, num_channels, seq_len = get_dls(args)
    print("Finished get_dls.")
    labels = np.stack([y.numpy() for x, y in dls.train.dataset])
    class_counts = labels.sum(axis=0)
    sample_weights = (1.0 / (class_counts + 1e-6))[np.newaxis, :] * labels
    sample_weights = sample_weights.sum(axis=1)
    sample_weights = np.power(sample_weights, oversample_power)
    sample_weights = sample_weights / sample_weights.sum()
    print("Class counts:", class_counts)
    print("Sample weights (first 10):", sample_weights[:10])
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    dls.train = DataLoader(
        dls.train.dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers
    )
    print("Oversampled DataLoader ready.")
    return dls, num_channels, seq_len

def finetune_func(lr=args.lr):
    print('Starting end-to-end finetuning')
    dls, num_channels, seq_len = get_dls_with_oversampling(args, oversample_power=2.0)
    print("Dataloaders with oversampling ready.")

    # Calculate class frequencies for ClassBalancedLoss
    labels = np.concatenate([y for x, y in dls.train])
    samples_per_cls = labels.sum(axis=0)
    print("Samples per class:", samples_per_cls)

    # Initialize ClassBalancedLoss
    loss_func = ClassBalancedFocalLoss(samples_per_cls, beta=0.9999, gamma=2.0)
    print("Using ClassBalancedLoss.")

    model = get_model(
        num_channels, args, head_type='classification', num_classes=len(dls.train.dataset.all_codes)
    )
    print("Model for finetuning ready.")
    model = transfer_weights(args.pretrained_model, model)
    print("Weights transferred.")

    cbs = [RevInCB(num_channels, denorm=True)] if args.revin else []
    cbs += [
        PatchCB(patch_len=args.patch_len, stride=args.stride),
        SaveModelCB(monitor='valid_loss', fname=args.save_finetuned_model, path=args.save_path)
    ]
    learn = Learner(
        dls, model,
        loss_func,
        lr=lr,
        cbs=cbs,
        metrics=[mse, mae, accuracy, precision, recall, f1]
    )
    print("Learner created. Starting fine-tune...")
    learn.fine_tune(n_epochs=args.n_epochs_finetune, base_lr=lr, freeze_epochs=10)
    print("Fine-tune complete.")
    save_recorders(learn)
    print("Recorder saved.")
    best_thrs = find_best_thresholds(learn, dls)
    np.save(args.save_path + args.save_finetuned_model + '_best_thrs.npy', best_thrs)
    print("Best thresholds saved.")
    return best_thrs

def linear_probe_func(lr=args.lr):
    print('Starting linear probing')
    dls, num_channels, seq_len = get_dls_with_oversampling(args, oversample_power=2.0)
    print("Dataloaders with oversampling ready.")
    labels = np.concatenate([y for x, y in dls.train])
    pos_weight = (labels.shape[0] - labels.sum(axis=0)) / (labels.sum(axis=0) + 1e-6)
    print("Positive weights calculated.")
    support = labels.sum(axis=0) 
    rare_classes = np.where(support <= 5)[0]
    for i in rare_classes:
        pos_weight[i] *= 5
    pos_weight = torch.tensor(pos_weight, dtype=torch.float32).to('cuda')
    print("Positive weights for rare classes boosted.")
    model = get_model(dls.vars, args, head_type='classification', num_classes=len(dls.train.dataset.all_codes))
    print("Model for linear probe ready.")
    model = transfer_weights(args.pretrained_model, model)
    print("Weights transferred.")
    loss_func = FocalLoss(alpha=1, gamma=2)
    cbs = [RevInCB(num_channels, denorm=True)] if args.revin else []
    cbs += [
         PatchCB(patch_len=args.patch_len, stride=args.stride),
         SaveModelCB(monitor='valid_loss', fname=args.save_finetuned_model, path=args.save_path)
        ]
    learn = Learner(
        dls, model, 
        loss_func, 
        lr=lr, 
        cbs=cbs,
        metrics=[mse, mae, accuracy, precision, recall, f1]
    )
    print("Learner created. Starting linear probe...")
    learn.linear_probe(n_epochs=args.n_epochs_finetune, base_lr=lr)
    print("Linear probe complete.")
    save_recorders(learn)
    print("Recorder saved.")
    best_thrs = find_best_thresholds(learn, dls)
    np.save(args.save_path + args.save_finetuned_model + '_best_thrs.npy', best_thrs)
    print("Best thresholds saved.")
    return best_thrs

def test_func(weight_path, best_thrs):
    print("Starting test_func...")
    args.root_path = "/home/paz3b/DeepPulse/src/data/datasets/physionet.org/files/ptb-xl/1.0.3/"
    args.data_path = "ptbxl_database_cleaned.csv"
    args.target = "scp_codes"
    print("Getting dataloaders for test...")
    dls, num_channels, seq_len = get_dls(args)
    print("Dataloaders ready.")
    num_classes = len(dls.train.dataset.all_codes)
    print("Creating model for test...")
    model = get_model(num_channels, args, head_type='classification', num_classes=num_classes).to('cuda')
    print("Model ready.")
    cbs = [RevInCB(num_channels, denorm=True)] if args.revin else []
    cbs += [PatchCB(patch_len=args.patch_len, stride=args.stride)]
    learn = Learner(dls, model, cbs=cbs, metrics=[mse, mae, accuracy, precision, recall, f1])
    print("Starting test...")
    weight_path = '/home/paz3b/DeepPulse/saved_models/PTB-XL/masked_patchtst/based_model/patchtst_pretrained_cw512_patch12_stride12_epochs-pretrain10_mask0.4_model1.pth'
    out  = learn.test(dls.test, weight_path=weight_path, scores=[mse, mae, accuracy, precision, recall, f1])
    print('score:', out[2])
    pd.DataFrame(
        np.array(out[2]).reshape(1, -1), 
        columns=['mse', 'mae', 'accuracy', 'precision', 'recall', 'f1']
    ).to_csv(
        args.save_path + args.save_finetuned_model + '_acc.csv', 
        float_format='%.6f', index=False
    )
    print("Test scores saved.")
    yt = np.concatenate([y for x, y in dls.test])
    probs = torch.sigmoid(torch.tensor(out[0])).numpy()
    print("Max probability per class:", probs.max(axis=0))
    support = yt.sum(axis=0)
    rare_class_indices = np.where(support <= 5)[0]
    global_threshold = 0.2
    yp = (probs > global_threshold).astype(int)
    for i in rare_class_indices:
        yp[:, i] = (probs[:, i] > 0.2).astype(int)
    yt = np.asarray(yt)
    yp = np.asarray(yp)
    if yt.ndim == 1 or (yt.ndim == 2 and yt.shape[1] == 1):
        yt = np.eye(yp.shape[1])[yt.reshape(-1)]
    if yp.ndim == 1 or (yp.ndim == 2 and yp.shape[1] == 1):
        yp = np.eye(yt.shape[1])[yp.reshape(-1)]
    yt = (yt > 0.5).astype(int)
    yp = (yp > 0.5).astype(int)
    report = classification_report(yt, yp, output_dict=True, zero_division=0)
    pd.DataFrame(report).to_csv(args.save_path + args.save_finetuned_model + '_per_class_report.csv')
    print("Classification report saved.")
    f1s = {k: v['f1-score'] for k, v in report.items() if k.isdigit()}
    precisions = {k: v['precision'] for k, v in report.items() if k.isdigit()}
    recalls = {k: v['recall'] for k, v in report.items() if k.isdigit()}
    print("Lowest F1 classes:", sorted(f1s.items(), key=lambda x: x[1])[:5])
    print("Lowest precision classes:", sorted(precisions.items(), key=lambda x: x[1])[:5])
    print("Lowest recall classes:", sorted(recalls.items(), key=lambda x: x[1])[:5])
    print("Highest F1 classes:", sorted(f1s.items(), key=lambda x: x[1], reverse=True)[:5])
    supports = {k: v['support'] for k, v in report.items() if k.isdigit()}
    print("Lowest support classes:", sorted(supports.items(), key=lambda x: x[1])[:5])
    print("Highest support classes:", sorted(supports.items(), key=lambda x: x[1], reverse=True)[:5])
    import scipy.stats
    support_vals = np.array([supports[k] for k in f1s])
    f1_vals = np.array([f1s[k] for k in f1s])
    if len(support_vals) > 1 and len(f1_vals) > 1:
        corr, pval = scipy.stats.pearsonr(support_vals, f1_vals)
        print(f"Correlation between support and F1: {corr:.3f} (p={pval:.3g})")
    low_support_f1 = {k: f1s[k] for k in f1s if supports[k] <= 5}
    print("F1 for classes with support <= 5:", low_support_f1)
    print("Test function complete.")
    return out

def mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    mixed_y = lam * y + (1 - lam) * y[index, :]
    return mixed_x, mixed_y

if __name__ == '__main__':
    print("Main script started")
    if args.is_finetune:
        print("Finetune mode")
        args.dset = args.dset_finetune
        args.root_path = "/home/paz3b/DeepPulse/src/data/datasets/physionet.org/files/ptb-xl/1.0.3/"
        args.data_path = "ptbxl_database_cleaned.csv"
        args.target = "scp_codes"
        suggested_lr = find_lr(head_type='classification')        
        best_thrs = finetune_func(suggested_lr)        
        print('finetune completed')
        out = test_func(args.save_path+args.save_finetuned_model, best_thrs)         
        print('----------- Complete! -----------')
    elif args.is_linear_probe:
        print("Linear probe mode")
        args.dset = args.dset_finetune
        suggested_lr = find_lr(head_type='classification')        
        best_thrs = linear_probe_func(suggested_lr)        
        print('finetune completed')
        out = test_func(args.save_path+args.save_finetuned_model, best_thrs)        
        print('----------- Complete! -----------')
    else:
        print("Test-only mode")
        args.dset = args.dset_finetune
        args.root_path = "/home/paz3b/DeepPulse/src/data/datasets/physionet.org/files/ptb-xl/1.0.3/"
        args.data_path = "ptbxl_database_cleaned.csv"
        args.target = "scp_codes"
        weight_path = args.save_path+args.dset_finetune+'_patchtst_finetuned'+suffix_name
        print("Getting dataloaders for test-only mode...")
        # Get number of classes from the dataset
        num_classes = len(get_dls(args)[0].train.dataset.all_codes)
        best_thrs = np.full(num_classes, 0.5)


report_path = args.save_path + args.save_finetuned_model + '_per_class_report.csv'
df = pd.read_csv(report_path, index_col=0)

# Only keep class columns (digits)
class_cols = [col for col in df.columns if col.isdigit()]
metrics = ['precision', 'recall', 'f1-score', 'support']

for metric in metrics:
    plt.figure(figsize=(16,4))
    plt.bar(class_cols, df.iloc[metric, class_cols].astype(float))
    plt.title(f'Per-class {metric}')
    plt.xlabel('Class')
    plt.ylabel(metric)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()