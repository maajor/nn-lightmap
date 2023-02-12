import math
import torch
from torch import nn
import torch.nn.functional as F
from siren_pytorch import Siren

class SirenGINet(nn.Module):
    def __init__(self):
        super().__init__()

        self.lm_layers = nn.ModuleList([])
        self.rf_layers = nn.ModuleList([])

        # l1
        self.lm_layers.append(Siren(dim_in=6, dim_out=64, w0=20, is_first=True))
        # l2, l3
        for _ in range(2):
            self.lm_layers.append(Siren(dim_in=64, dim_out=64))
        # l4
        self.lm_layers.append(Siren(dim_in=64, dim_out=32))
        # view direction input up dim
        self.vup_layers = Siren(dim_in=3, dim_out=32, w0=60, is_first=True)  # v input
        # rf layers
        for _ in range(3):
            self.rf_layers.append(Siren(dim_in=64, dim_out=64))
        # l8
        self.rf_layers.append(Siren(dim_in=64, dim_out=3, activation=nn.Identity()))

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
