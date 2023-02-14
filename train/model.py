import math
import torch
from torch import nn
import torch.nn.functional as F
from siren_pytorch import Siren


class SirenGINet(nn.Module):
    def __init__(self, dim_hidden=64, num_layer=2):
        super().__init__()

        self.lm_layers = nn.ModuleList([])
        self.rf_layers = nn.ModuleList([])

        # l1
        self.lm_layers.append(Siren(dim_in=6, dim_out=dim_hidden, w0=20, is_first=True))
        # l2, l3
        for _ in range(num_layer):
            self.lm_layers.append(Siren(dim_in=dim_hidden, dim_out=dim_hidden))
        # l4
        self.lm_layers.append(Siren(dim_in=dim_hidden, dim_out=32))
        # view direction input up dim
        self.vup_layers = Siren(dim_in=3, dim_out=32, w0=60, is_first=True)  # v input
        # rf layers
        self.rf_layers.append(Siren(dim_in=64, dim_out=dim_hidden))
        for _ in range(num_layer):
            self.rf_layers.append(Siren(dim_in=dim_hidden, dim_out=dim_hidden))
        # l8
        self.rf_layers.append(
            Siren(dim_in=dim_hidden, dim_out=3, activation=nn.Identity())
        )

    def forward(self, pn, v):
        x = pn
        # lightmap phase
        for layer in self.lm_layers:
            x = layer(x)

        # input view direction and concat
        vup = self.vup_layers(v)
        x = torch.cat((x, vup), -1)

        # refinement phase
        for layer in self.rf_layers:
            x = layer(x)

        return x
