import torch
import torch.nn as nn
from .core import Callback
from src.models.layers.revin import RevIN

class RevInCB(Callback):
    def __init__(self, num_features: int, eps=1e-5, 
                        affine:bool=False, denorm:bool=True):
        """        
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        :param denorm: if True, the output will be de-normalized

        This callback only works with affine=False.
        if affine=True, the learnable affine_weights and affine_bias are not learnt
        """
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.denorm = denorm
        self.revin = RevIN(num_features, eps, affine)
    

    def before_forward(self): self.revin_norm()
    def after_forward(self): 
        if self.denorm: self.revin_denorm() 
        
    def revin_norm(self):
        xb_revin = self.revin(self.xb, 'norm')      # xb_revin: [bs x seq_len x nvars]
        self.learner.xb = xb_revin

    # In src/callback/transforms.py
    def revin_denorm(self):
        print(f"[DEBUG] Tensor shape before RevIN: {self.pred.shape}")

        # --- Skip all sequence padding/slicing for classification/multilabel ---
        if self.pred.ndim == 2:
            print("[INFO] Skipping sequence padding/slicing and RevIN denorm for classification/multilabel output.")
            self.preds = self.pred.contiguous()
            self.targets = self.learner.yb.contiguous()
            print(f"[DEBUG] Final pred shape: {self.preds.shape}, Final targ shape: {self.targets.shape}")
            assert self.preds.shape == self.targets.shape, f"Shape mismatch: preds {self.preds.shape}, targets {self.targets.shape}"
            return
        # Defensive: If pred batch size does not match yb, skip or fix
        if hasattr(self, 'learner') and hasattr(self.learner, 'yb') and self.learner.yb is not None:
            targ = self.learner.yb
            if self.pred.size(0) != targ.shape[0]:
                print(f"[ERROR] Prediction batch size {self.pred.size(0)} does not match target batch size {targ.shape[0]}. Slicing predictions to match targets.")
                self.pred = self.pred[:targ.shape[0]]
                batch_size = self.pred.size(0)
            else:
                batch_size = self.pred.size(0)
        target_seq_len = 416

        if hasattr(self, 'learner') and hasattr(self.learner, 'yb') and self.learner.yb is not None:
            targ = self.learner.yb

        # Handle 4D tensor (batch, patches, patch_size, features)
        if self.pred.ndim == 4:
            num_patches, patch_size, num_features = self.pred.shape[1], self.pred.shape[2], self.pred.shape[3]
            print(f"[DEBUG] Original shape: [batch_size={batch_size}, num_patches={num_patches}, patch_size={patch_size}, num_features={num_features}]")
            sequence_length = num_patches * patch_size
            self.pred = self.pred.reshape(batch_size, sequence_length, num_features)
            if num_features != self.num_features:
                features_per_group = num_features // self.num_features
                usable_features = features_per_group * self.num_features
                if usable_features != num_features:
                    print(f"[WARNING] Truncating features from {num_features} to {usable_features} to match self.num_features={self.num_features}")
                    self.pred = self.pred[:, :, :usable_features]
                self.pred = self.pred.reshape(batch_size, sequence_length, self.num_features, features_per_group)
                self.pred = self.pred.mean(dim=-1)
            print(f"[DEBUG] Shape before RevIN: {self.pred.shape}")

        # Ensure consistent sequence length for all batches
        if self.pred.shape[1] > target_seq_len:
            print(f"[WARNING] Slicing predictions from {self.pred.shape[1]} to {target_seq_len} to match targets")
            self.pred = self.pred[:, :target_seq_len, :]
        elif self.pred.shape[1] < target_seq_len:
            print(f"[WARNING] Padding predictions from {self.pred.shape[1]} to {target_seq_len} to match targets")
            pad = target_seq_len - self.pred.shape[1]
            self.pred = torch.nn.functional.pad(self.pred, (0, 0, 0, pad))

        # --- Ensure targets are same shape and batch size as predictions ---
        if hasattr(self, 'learner') and hasattr(self.learner, 'yb') and self.learner.yb is not None:
            targ = self.learner.yb

            # Slice/pad targets along sequence dimension
            if targ.ndim == 3 and targ.shape[1] > target_seq_len:
                print(f"[WARNING] Slicing targets from {targ.shape[1]} to {target_seq_len} to match predictions")
                targ = targ[:, :target_seq_len, :]
            elif targ.ndim == 3 and targ.shape[1] < target_seq_len:
                print(f"[WARNING] Padding targets from {targ.shape[1]} to {target_seq_len} to match predictions")
                pad = target_seq_len - targ.shape[1]
                targ = torch.nn.functional.pad(targ, (0, 0, 0, pad))

            # Ensure batch size matches
            if targ.shape[0] != batch_size:
                print(f"[WARNING] Target batch size {targ.shape[0]} does not match prediction batch size {batch_size}. Using current batch size.")
                if targ.shape[0] > batch_size:
                    targ = targ[:batch_size]
                else:
                    raise RuntimeError(f"Target batch size {targ.shape[0]} is less than prediction batch size {batch_size}.")

            # Expand targets if needed
            if targ.ndim == 1 and self.pred.ndim == 3:
                print(f"[WARNING] Expanding targets from {targ.shape} to {self.pred.shape} for loss")
                targ = targ.unsqueeze(-1).unsqueeze(-1).expand(batch_size, self.pred.shape[1], self.pred.shape[2])

            self.learner.yb = targ

            # Ensure RevIN stats match current batch size
            if hasattr(self.revin, "mean") and self.revin.mean is not None:
                if self.revin.mean.size(0) != batch_size:
                    print(f"[WARNING] Adjusting RevIN mean from batch {self.revin.mean.size(0)} to {batch_size}")
                    self.revin.mean = self.revin.mean[:1].expand(batch_size, -1, -1).contiguous()
            if hasattr(self.revin, "stdev") and self.revin.stdev is not None:
                if self.revin.stdev.size(0) != batch_size:
                    print(f"[WARNING] Adjusting RevIN stdev from batch {self.revin.stdev.size(0)} to {batch_size}")
                    self.revin.stdev = self.revin.stdev[:1].expand(batch_size, -1, -1).contiguous()

        # --- Defensive: ensure targets shape matches predictions exactly ---
        if self.learner.yb.shape != self.pred.shape:
            print(f"[ERROR] Shape mismatch before loss: pred {self.pred.shape}, targ {self.learner.yb.shape}")
            # Try to permute if axes are swapped
            if self.learner.yb.shape == (self.pred.shape[0], self.pred.shape[2], self.pred.shape[1]):
                self.learner.yb = self.learner.yb.permute(0, 2, 1)
                print(f"[INFO] Permuted targets to {self.learner.yb.shape}")
            elif self.learner.yb.shape == (self.pred.shape[1], self.pred.shape[0], self.pred.shape[2]):
                self.learner.yb = self.learner.yb.permute(1, 0, 2)
                print(f"[INFO] Permuted targets to {self.learner.yb.shape}")
            else:
                raise RuntimeError(f"Cannot automatically align target shape {self.learner.yb.shape} to prediction shape {self.pred.shape}")

        # --- Final axis check and fix for scoring ---
        # If axes are still swapped, permute so both are [batch, seq, features]
        if self.pred.shape[1] != 416 and self.pred.shape[2] == 416:
            print("[INFO] Swapping axes 1 and 2 for preds")
            self.pred = self.pred.permute(0, 2, 1).contiguous()
        if self.learner.yb.shape[1] != 416 and self.learner.yb.shape[2] == 416:
            print("[INFO] Swapping axes 1 and 2 for targets")
            self.learner.yb = self.learner.yb.permute(0, 2, 1).contiguous()

        # --- Apply RevIN denormalization ---
        pred = self.revin(self.pred, 'denorm')
        print(f"[DEBUG] Shape after RevIN: {pred.shape}")
        self.pred = pred

        if self.pred.ndim == 3:
            print(f"[DEBUG] 3D pred shape for loss: {self.pred.shape}")

        # --- Now update scoring tensors ---
        self.preds = self.pred.contiguous()
        self.targets = self.learner.yb.contiguous()
        print(f"[DEBUG] Final pred shape: {self.preds.shape}, Final targ shape: {self.targets.shape}")

        # Final safety: check axes
        assert self.preds.shape == self.targets.shape, f"Shape mismatch: preds {self.preds.shape}, targets {self.targets.shape}"
        # Check axes order
        if self.preds.shape[1] != self.targets.shape[1]:
            print(f"[ERROR] Sequence axis mismatch: preds.shape[1]={self.preds.shape[1]}, targets.shape[1]={self.targets.shape[1]}")
        if self.preds.shape[2] != self.targets.shape[2]:
            print(f"[ERROR] Feature axis mismatch: preds.shape[2]={self.preds.shape[2]}, targets.shape[2]={self.targets.shape[2]}")
        # Print a sample slice to check axes
        print(f"[DEBUG] preds[0,0,:10]: {self.preds[0,0,:10]}")
        print(f"[DEBUG] targets[0,0,:10]: {self.targets[0,0,:10]}")
    
    def restore_shape(self):
        """Restore tensor to original shape if needed"""
        if hasattr(self, 'original_shape') and self.original_shape is not None:
            batch_size, num_patches, patch_size, num_features = self.original_shape
            sequence_length = self.pred.size(1)
            if sequence_length == num_patches * patch_size:
                self.pred = self.pred.view(batch_size, patch_size, num_patches, num_features)
                self.pred = self.pred.permute(0, 2, 1, 3)  # [batch_size, num_patches, patch_size, num_features]