import math
import torch
from torch import nn
import torch.nn.functional as F
from siren_pytorch import Siren


class SirenGINet(nn.Module):
    def __init__(self, lm_dim=256, lm_layer=5, dim_hidden=32, rf_dim=64, rf_layer=3):
        super().__init__()

        self.lm_layers = nn.ModuleList([])
        self.rf_layers = nn.ModuleList([])

        # l1
        self.lm_layers.append(Siren(dim_in=6, dim_out=lm_dim, w0=20, is_first=True))
        # l2, l3
        for _ in range(lm_layer):
            self.lm_layers.append(Siren(dim_in=lm_dim, dim_out=lm_dim))
        # l4
        self.lm_layers.append(Siren(dim_in=lm_dim, dim_out=dim_hidden))
        # view direction input up dim
        self.vup_layers = Siren(dim_in=3, dim_out=dim_hidden, w0=60, is_first=True)  # v input
        # rf layers
        self.rf_layers.append(Siren(dim_in=dim_hidden*2, dim_out=rf_dim))
        for _ in range(rf_layer):
            self.rf_layers.append(Siren(dim_in=rf_dim, dim_out=rf_dim))
        # l8
        self.rf_layers.append(
            Siren(dim_in=rf_dim, dim_out=3, activation=nn.Sigmoid())
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

    def bake_lightmap(self, pn):
        x = pn
        # lightmap phase
        for layer in self.lm_layers:
            x = layer(x)

        return x

    def inference_with_lightmap(self, lightmap, v):
        vup = self.vup_layers(v)
        x = torch.cat((lightmap, vup), -1)

        # refinement phase
        for layer in self.rf_layers:
            x = layer(x)

        return x
