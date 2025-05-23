

import numpy as np
import pandas as pd
import os
import torch
from torch import nn

from src.models.patchTST import PatchTST
from src.learner import Learner, transfer_weights
from src.callback.tracking import *
from src.callback.patch_mask import *
from src.callback.transforms import *
from src.metrics import *
from src.basics import set_device
from datautils import *


import argparse

parser = argparse.ArgumentParser()
# Dataset and dataloader
parser = argparse.ArgumentParser()
# Dataset and dataloader
parser.add_argument('--dset_pretrain', type=str, default='PTB-XL', help='dataset name')
parser.add_argument('--context_points', type=int, default=512, help='sequence length')
parser.add_argument('--target_points', type=int, default=96, help='forecast horizon')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--num_workers', type=int, default=0, help='number of workers for DataLoader')
parser.add_argument('--scaler', type=str, default='standard', help='scale the input data')
parser.add_argument('--features', type=str, default='M', help='for multivariate model or univariate model')
# Dataset paths
parser.add_argument('--root_path', type=str, default='/home/paz3b/DeepPulse/src/data/datasets/physionet.org/files/ptb-xl/1.0.3/', help='Root path to the dataset')
parser.add_argument('--data_path', type=str, default='ptbxl_database_cleaned.csv', help='Path to the dataset file')
parser.add_argument('--target', type=str, default='filename_hr', help='Target column in the dataset')
# Patch
parser.add_argument('--patch_len', type=int, default=12, help='patch length')
parser.add_argument('--stride', type=int, default=12, help='stride between patch')
# RevIN
parser.add_argument('--revin', type=int, default=1, help='reversible instance normalization')
# Model args
parser.add_argument('--n_layers', type=int, default=3, help='number of Transformer layers')
parser.add_argument('--n_heads', type=int, default=16, help='number of Transformer heads')
parser.add_argument('--d_model', type=int, default=128, help='Transformer d_model')
parser.add_argument('--d_ff', type=int, default=512, help='Tranformer MLP dimension')
parser.add_argument('--dropout', type=float, default=0.2, help='Transformer dropout')
parser.add_argument('--head_dropout', type=float, default=0.2, help='head dropout')
# Pretrain mask
parser.add_argument('--mask_ratio', type=float, default=0.4, help='masking ratio for the input')
# Optimization args
parser.add_argument('--n_epochs_pretrain', type=int, default=10, help='number of pre-training epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
# model id to keep track of the number of models saved
parser.add_argument('--pretrained_model_id', type=int, default=1, help='id of the saved pretrained model')
parser.add_argument('--model_type', type=str, default='based_model', help='for multivariate model or univariate model')


args = parser.parse_args()
print('args:', args)
args.save_pretrained_model = 'patchtst_pretrained_cw'+str(args.context_points)+'_patch'+str(args.patch_len) + '_stride'+str(args.stride) + '_epochs-pretrain' + str(args.n_epochs_pretrain) + '_mask' + str(args.mask_ratio)  + '_model' + str(args.pretrained_model_id)
args.save_path = 'saved_models/' + args.dset_pretrain + '/masked_patchtst/' + args.model_type + '/'
if not os.path.exists(args.save_path): os.makedirs(args.save_path)


# get available GPU devide
set_device()


def get_model(c_in, args):
    """
    c_in: number of variables
    """
    # Force num_patch to a specific value
    num_patch = 416  # Set this to the desired value (e.g., 42 or 416)
    print(f"[INFO] Forcing number of patches to: {num_patch}")

    # Initialize the model
    model = PatchTST(c_in=c_in,
                     target_dim=args.target_points,
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
                     head_type='pretrain',
                     res_attention=False)
    print('number of model params', sum(p.numel() for p in model.parameters() if p.requires_grad))
    return model
    
def load_checkpoint(model, checkpoint_path):
    """
    Load a checkpoint and handle mismatched parameters.
    """
    checkpoint = torch.load(checkpoint_path)
    state_dict = checkpoint["state_dict"]

    # Debug: Print shapes of parameters in the checkpoint and the model
    print("[DEBUG] Checking parameter shapes:")
    for name, param in state_dict.items():
        if name in model.state_dict():
            model_param = model.state_dict()[name]
            if param.shape != model_param.shape:
                print(f"[DEBUG] Shape mismatch for {name}: checkpoint {param.shape}, model {model_param.shape}")
        else:
            print(f"[DEBUG] {name} not found in model.")

    # Remove mismatched parameters
    for name in list(state_dict.keys()):
        if name in model.state_dict() and state_dict[name].shape != model.state_dict()[name].shape:
            print(f"[INFO] Ignoring {name} due to shape mismatch.")
            del state_dict[name]

    # Load the state dictionary into the model
    model.load_state_dict(state_dict, strict=False)
    print("[INFO] Checkpoint loaded successfully.")
    return model

def get_checkpoint_num_patch(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    if "W_pos" in checkpoint["state_dict"]:
        checkpoint_num_patch = checkpoint["state_dict"]["W_pos"].shape[1]
        print(f"[INFO] Number of patches in checkpoint: {checkpoint_num_patch}")
        return checkpoint_num_patch
    else:
        raise ValueError("W_pos not found in checkpoint.")

def find_lr():
    # Get DataLoader and metadata
    train_loader, num_channels, seq_len = get_dls(args)  # Ensure get_dls returns num_channels

    # Initialize the model
    model = get_model(num_channels, args)  # Pass num_channels to get_model

    # Define the loss function
    loss_func = torch.nn.MSELoss(reduction='mean')

    # Define callbacks
    cbs = [RevInCB(num_channels, denorm=False)] if args.revin else []
    cbs += [PatchMaskCB(patch_len=args.patch_len, stride=args.stride, d_model=args.d_model, mask_ratio=args.mask_ratio)]

    # Define the learner
    learn = Learner(train_loader, model, loss_func, lr=args.lr, cbs=cbs)

    # Find the learning rate
    suggested_lr = learn.lr_finder()
    print('Suggested Learning Rate:', suggested_lr)
    return suggested_lr


def pretrain_func(lr=args.lr):
    # Skip checkpoint logic
    args.context_points = 5000
    args.patch_len = 12
    args.stride = 12
    print(f"[INFO] Updated args: context_points={args.context_points}, patch_len={args.patch_len}, stride={args.stride}")

    # Get DataLoader and metadata
    dls, num_channels, seq_len = get_dls(args)

    # Set the vars attribute for compatibility
    dls.vars = num_channels

    # Initialize the model
    model = get_model(num_channels, args)

    # Define the loss function
    loss_func = torch.nn.MSELoss(reduction='mean')

    # Define callbacks
    cbs = [RevInCB(dls.vars, denorm=False)] if args.revin else []  # Use dls.vars here
    cbs += [
        PatchMaskCB(patch_len=args.patch_len, stride=args.stride, d_model=args.d_model, mask_ratio=args.mask_ratio),
        SaveModelCB(monitor='valid_loss', fname=args.save_pretrained_model, path=args.save_path)
    ]

    # Define the learner
    learn = Learner(dls, model, loss_func, lr=lr, cbs=cbs)

    # Fit the data to the model
    learn.fit_one_cycle(n_epochs=args.n_epochs_pretrain, lr_max=lr)

    train_loss = learn.recorder['train_loss']
    valid_loss = learn.recorder['valid_loss']
    df = pd.DataFrame(data={'train_loss': train_loss, 'valid_loss': valid_loss})
    df.to_csv(args.save_path + args.save_pretrained_model + '_losses.csv', float_format='%.6f', index=False)


if __name__ == '__main__':
    
    args.dset = args.dset_pretrain
    suggested_lr = find_lr()
    # Pretrain
    pretrain_func(suggested_lr)
    print('pretraining completed')
