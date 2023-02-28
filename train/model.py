import math
import torch
from torch import nn
from siren_pytorch import Siren
import numpy as np
import re

BOX_OFFSETS = torch.tensor([[i,j] for i in [0, 1] for j in [0, 1]], device='cuda')

def hash(coords, log2_hashmap_size):
    '''
    coords: this function can process upto 7 dim coordinates
    log2T:  logarithm of T w.r.t 2
    '''
    primes = [1, 2654435761, 805459861, 3674653429, 2097192037, 1434869437, 2165219737]

    xor_result = torch.zeros_like(coords)[..., 0]
    for i in range(coords.shape[-1]):
        xor_result ^= coords[..., i]*primes[i]

    return torch.tensor((1<<log2_hashmap_size)-1).to(xor_result.device) & xor_result

def get_grid_indices(coords, resolution):
    return coords[...,0] * resolution + coords[...,1]

class SHEncoder(nn.Module):
    def __init__(self, input_dim=3, degree=4):
    
        super().__init__()

        self.input_dim = input_dim
        self.degree = degree

        assert self.input_dim == 3
        assert self.degree >= 1 and self.degree <= 5

        self.out_dim = degree ** 2

        self.C0 = 0.28209479177387814
        self.C1 = 0.4886025119029199
        self.C2 = [
            1.0925484305920792,
            -1.0925484305920792,
            0.31539156525252005,
            -1.0925484305920792,
            0.5462742152960396
        ]
        self.C3 = [
            -0.5900435899266435,
            2.890611442640554,
            -0.4570457994644658,
            0.3731763325901154,
            -0.4570457994644658,
            1.445305721320277,
            -0.5900435899266435
        ]
        self.C4 = [
            2.5033429417967046,
            -1.7701307697799304,
            0.9461746957575601,
            -0.6690465435572892,
            0.10578554691520431,
            -0.6690465435572892,
            0.47308734787878004,
            -1.7701307697799304,
            0.6258357354491761
        ]

    def forward(self, input, **kwargs):

        result = torch.empty((*input.shape[:-1], self.out_dim), dtype=input.dtype, device=input.device)
        x, y, z = input.unbind(-1)

        result[..., 0] = self.C0
        if self.degree > 1:
            result[..., 1] = -self.C1 * y
            result[..., 2] = self.C1 * z
            result[..., 3] = -self.C1 * x
            if self.degree > 2:
                xx, yy, zz = x * x, y * y, z * z
                xy, yz, xz = x * y, y * z, x * z
                result[..., 4] = self.C2[0] * xy
                result[..., 5] = self.C2[1] * yz
                result[..., 6] = self.C2[2] * (2.0 * zz - xx - yy)
                #result[..., 6] = self.C2[2] * (3.0 * zz - 1) # xx + yy + zz == 1, but this will lead to different backward gradients, interesting...
                result[..., 7] = self.C2[3] * xz
                result[..., 8] = self.C2[4] * (xx - yy)
                if self.degree > 3:
                    result[..., 9] = self.C3[0] * y * (3 * xx - yy)
                    result[..., 10] = self.C3[1] * xy * z
                    result[..., 11] = self.C3[2] * y * (4 * zz - xx - yy)
                    result[..., 12] = self.C3[3] * z * (2 * zz - 3 * xx - 3 * yy)
                    result[..., 13] = self.C3[4] * x * (4 * zz - xx - yy)
                    result[..., 14] = self.C3[5] * z * (xx - yy)
                    result[..., 15] = self.C3[6] * x * (xx - 3 * yy)
                    if self.degree > 4:
                        result[..., 16] = self.C4[0] * xy * (xx - yy)
                        result[..., 17] = self.C4[1] * yz * (3 * xx - yy)
                        result[..., 18] = self.C4[2] * xy * (7 * zz - 1)
                        result[..., 19] = self.C4[3] * yz * (7 * zz - 3)
                        result[..., 20] = self.C4[4] * (zz * (35 * zz - 30) + 3)
                        result[..., 21] = self.C4[5] * xz * (7 * zz - 3)
                        result[..., 22] = self.C4[6] * (xx - yy) * (7 * zz - 1)
                        result[..., 23] = self.C4[7] * xz * (xx - 3 * yy)
                        result[..., 24] = self.C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy))

        return result
    
    def dump_code(self, base=4):
        lines = []
        lines.append(f'vec4 f0{0+base} = vec4({"{0:.3f}".format(self.C0)},-{"{0:.3f}".format(self.C1)}*dir.y,{"{0:.3f}".format(self.C1)}*dir.z,-{"{0:.3f}".format(self.C1)}*dir.x);')
        lines.append(f'vec3 xxyyzz=dir*dir;')
        lines.append(f'vec3 xyyzxz=dir.xyx*dir.yzz;')
        lines.append(f'vec4 f0{1+base}=vec4({"{0:.3f}".format(self.C2[0])}*xyyzxz.x,{"{0:.3f}".format(self.C2[1])}*xyyzxz.y,{self.C2[2]}*(2.f*xxyyzz.z-xxyyzz.x-xxyyzz.y),{"{0:.3f}".format(self.C2[3])}*xyyzxz.z);')
        lines.append(f'vec4 f0{2+base}=vec4({"{0:.3f}".format(self.C2[4])}*(xxyyzz.x-xxyyzz.y),{"{0:.3f}".format(self.C3[0])}*dir.y*(3.f*xxyyzz.x-xxyyzz.y),{"{0:.3f}".format(self.C3[1])}*xyyzxz.x*dir.z,{"{0:.3f}".format(self.C3[2])}*dir.y*(4.f*xxyyzz.z-xxyyzz.x-xxyyzz.y));')
        lines.append(f'vec4 f0{3+base}=vec4({"{0:.3f}".format(self.C3[3])}*dir.z*(2.f*xxyyzz.z-3.f*xxyyzz.x-3.f*xxyyzz.y), {"{0:.3f}".format(self.C3[4])}*dir.x*(4.f*xxyyzz.z-xxyyzz.x-xxyyzz.y), {"{0:.3f}".format(self.C3[5])}*dir.z*(xxyyzz.x-xxyyzz.y), {"{0:.3f}".format(self.C3[6])}*dir.x*(xxyyzz.x-3.f*xxyyzz.y));')
        return lines


class SirenGINet(nn.Module):
    def __init__(self, lm_dim=64, lm_layer=2, dim_hidden=16, rf_dim=32, rf_layer=2):
        super().__init__()

        self.lm_layers = nn.ModuleList([])
        self.rf_layers = nn.ModuleList([])
        self.sh_encoder = SHEncoder(degree=4)
        self.sh_dim = 16 # degree 4 sh
        self.cs_level = 8
        self.dim_lightmap = dim_hidden

        self.lm_layers.append(Siren(dim_in=self.cs_level*6 + self.sh_dim, dim_out=lm_dim, w0=20, is_first=True))

        for _ in range(lm_layer):
            self.lm_layers.append(Siren(dim_in=lm_dim, dim_out=lm_dim))

        self.lm_layers.append(Siren(dim_in=lm_dim, dim_out=dim_hidden))
        
        # rf layers
        self.rf_layers.append(Siren(dim_in=dim_hidden + self.sh_dim, dim_out=rf_dim))
        for _ in range(rf_layer):
            self.rf_layers.append(Siren(dim_in=rf_dim, dim_out=rf_dim))

        self.rf_layers.append(Siren(dim_in=rf_dim, dim_out=3, activation=nn.Sigmoid()))

    def forward(self, p, n, v):
        '''
        input: 
          uv: B x W x H x 2, 
          n: B x W x H x 3,
          v: B x W x H x 3
        '''
        n_in_sh = self.sh_encoder(n)

        cs_enc = [n_in_sh]
        for i in range(self.cs_level):
            s = torch.sin(2**i * p)
            c = torch.cos(2**i * p)
            cs_enc.append(s)
            cs_enc.append(c)
            
        x = torch.cat(cs_enc, dim=-1)

        # lightmap phase
        for layer in self.lm_layers:
            x = layer(x)

        # input view direction and concat
        v_in_sh = self.sh_encoder(v)
        x = torch.cat((x, v_in_sh), -1)

        # refinement phase
        for layer in self.rf_layers:
            x = layer(x)

        return x

    def bake_lightmap(self, p, n):
        n_in_sh = self.sh_encoder(n)

        cs_enc = [n_in_sh]
        for i in range(self.cs_level):
            s = torch.sin(2**i * p)
            c = torch.cos(2**i * p)
            cs_enc.append(s)
            cs_enc.append(c)
            
        x = torch.cat(cs_enc, dim=-1)

        # lightmap phase
        for layer in self.lm_layers:
            x = layer(x)

        return x

    def inference_with_lightmap(self, lightmap, v):
        v_in_sh = self.sh_encoder(v)
        x = torch.cat((lightmap, v_in_sh), -1)

        # refinement phase
        for layer in self.rf_layers:
            x = layer(x)

        return x

    def debug_shader(self, lm, v):
        v_in_sh = self.sh_encoder(v)
        x = torch.cat((lm, v_in_sh), -1)
        x = self.rf_layers[0](x)
        return x

    def dump_shader(self):

        def vec4(n):
            return 'vec4(' + ','.join(["{0:.3f}".format(i) for i in n.flatten()]) + ')'

        def mat4(n, transpose=False):
            if transpose: n = np.transpose(n)
            return 'mat4(' + ','.join(["{0:.3f}".format(i) for i in n.flatten()]) + ')'

        def vname(layer, chunk):
            return "f%d%d" % (layer, chunk)

        '''output_dim, input_dim = self.vup_layers.weight.shape
        w0 = self.vup_layers.activation.w0

        weights = self.vup_layers.weight.detach().numpy()
        bias = self.vup_layers.bias.detach().numpy()'''

        vec4_defs = []
        vec4_defs.extend(self.sh_encoder.dump_code(self.dim_lightmap // 4))

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
        vec4_defs.append('vec3 color = vec3(1.0) / ( vec3(1.0) + pow(vec3(2.71828), -outc.xyz));//sigmoid')

        out = '\n'.join(vec4_defs) + "\n"

        out = re.sub(r"(\d+\.\d*)0+\b", r"\1", out) # Remove trailing zeros eg. 1.0 => 1.
        out = re.sub(r"\b(\.\d+)0+\b", r"\1", out) # Remove trailing zeros eg. .60 => .6
        out = re.sub(r"\b0(\.\d+)\b", r"\1", out) # Remove leading zeros eg. 0.5 => .5
        out = re.sub(r"-\.0+\b", r".0", out) # Make all zeros positive eg. -.0 => .0
        out = re.sub(r"\+-", r"-", out) # Change +-1. into -1.

        print(out)

        return out