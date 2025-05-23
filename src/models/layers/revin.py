import torch
from torch import nn

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        print(f"[DEBUG] RevIN input shape: {x.shape}, mode: {mode}")
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        print(f"[DEBUG] RevIN output shape: {x.shape}")
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        self.mean = x.mean(dim=1, keepdim=True)
        self.stdev = x.std(dim=1, keepdim=True)
        x = (x - self.mean) / (self.stdev + 1e-5)
        return x

    def _denormalize(self, x):
        print(f"[DEBUG] _denormalize input shape: {x.shape}")
        print(f"[DEBUG] self.stdev shape: {self.stdev.shape}")
        print(f"[DEBUG] self.mean shape: {self.mean.shape}")

        # Handle additional dimensions (e.g., patches)
        if x.shape[-1] != self.stdev.shape[-1]:
            # Reshape or broadcast self.stdev and self.mean to match x
            stdev = self.stdev.expand(x.size(0), x.size(1), self.stdev.size(-1))
            mean = self.mean.expand(x.size(0), x.size(1), self.mean.size(-1))
        else:
            stdev = self.stdev
            mean = self.mean

        x = x * stdev + mean
        return x