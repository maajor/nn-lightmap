import math
import torch
from torch import nn
from siren_pytorch import Siren
import numpy as np
import re


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
        self.vup_layers = Siren(
            dim_in=3, dim_out=dim_hidden, w0=60, is_first=True
        )  # v input
        # rf layers
        self.rf_layers.append(Siren(dim_in=dim_hidden * 2, dim_out=rf_dim))
        for _ in range(rf_layer):
            self.rf_layers.append(Siren(dim_in=rf_dim, dim_out=rf_dim))
        # l8
        self.rf_layers.append(Siren(dim_in=rf_dim, dim_out=3, activation=nn.Sigmoid()))

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

    def debug_shader(self, lm, v):
        vup = self.vup_layers(v)
        x = torch.cat((lm, vup), -1)
        x = self.rf_layers[0](x)
        return x

    def dump_shader(self):

        def vec4(n):
            return 'vec4(' + ','.join(['{0:.3f}'.format(i) for i in n.flatten()]) + ')'

        def mat4(n, transpose=False):
            if transpose: n = np.transpose(n)
            return 'mat4(' + ','.join(['{0:.3f}'.format(i) for i in n.flatten()]) + ')'

        def vname(layer, chunk):
            return "f%d%d" % (layer, chunk)

        output_dim, input_dim = self.vup_layers.weight.shape
        w0 = self.vup_layers.activation.w0

        weights = self.vup_layers.weight.detach().numpy()
        bias = self.vup_layers.bias.detach().numpy()

        vec4_defs = ["vec4 x=vec4(view_dir,1);"]
        for chunk in range(output_dim // 4):
            mat = np.concatenate([
                w0*weights[chunk*4],[0],
                w0*weights[chunk*4+1],[0],
                w0*weights[chunk*4+2],[0],
                w0*weights[chunk*4+3],[0],
            ]).reshape(4,4)
            vec4_defs.append(
                'vec4 {}=sin(x*{}+{});'.format(
                    vname(0,chunk+output_dim/4), # lightmap has output_dim channels
                    mat4(mat, transpose=False),
                    vec4(w0*bias[chunk*4:chunk*4+4])
                )
            )

        layers = len(self.rf_layers)
        for layer in range(layers-1):
            w0=1.0
            weights = self.rf_layers[layer].weight.detach().numpy()
            bias = self.rf_layers[layer].bias.detach().numpy()
            output_dim, input_dim = weights.shape
            assert(input_dim % 4 == 0)
            assert(output_dim % 4 == 0)
            for out_chunk in range(output_dim // 4):
                elements = []
                for in_chunk in range(input_dim // 4):
                    elements.append(
                        mat4(w0*weights[out_chunk*4:4+out_chunk*4, in_chunk*4:4+in_chunk*4], transpose=True)
                        + '*'
                        + vname(layer, in_chunk)
                    )
                elements.append(
                    vec4(w0*bias[out_chunk*4:4+out_chunk*4])
                )
                vec4_defs.append(
                    'vec4 {}=sin({});'.format(
                        vname(layer+1, out_chunk),
                        '+'.join(elements)
                    )
                )

        # last layer export 3 dim
        output_dim, input_dim = self.rf_layers[-1].weight.shape

        weights = self.rf_layers[-1].weight.detach().numpy()
        bias = self.rf_layers[-1].bias.detach().numpy()
        elements = []
        for in_chunk in range(input_dim // 4):
            mat = np.concatenate([
                weights[0, in_chunk*4:4+in_chunk*4],
                weights[1, in_chunk*4:4+in_chunk*4],
                weights[2, in_chunk*4:4+in_chunk*4],
                [0,0,0,0]
            ]).reshape(4,4)
            elements.append(
                "{}*{}".format(
                    vname(layers - 1, in_chunk),
                    mat4(mat, transpose=False),
                )
            )
        elements.append(
                vec4(np.concatenate([bias, [0]]))
            )
        vec4_defs.append('vec4 outc={};'.format('+'.join(elements)))

        out = '\n'.join(vec4_defs) + "\n"

        out = re.sub(r"(\d+\.\d*)0+\b", r"\1", out) # Remove trailing zeros eg. 1.0 => 1.
        out = re.sub(r"\b(\.\d+)0+\b", r"\1", out) # Remove trailing zeros eg. .60 => .6
        out = re.sub(r"\b0(\.\d+)\b", r"\1", out) # Remove leading zeros eg. 0.5 => .5
        out = re.sub(r"-\.0+\b", r".0", out) # Make all zeros positive eg. -.0 => .0
        out = re.sub(r"\+-", r"-", out) # Change +-1. into -1.

        print(out)

        return out