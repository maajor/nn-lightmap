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

class SirenGINet(nn.Module):
    def __init__(self, lm_dim=64, lm_layer=2, dim_hidden=16, rf_dim=32, rf_layer=2):
        super().__init__()

        self.lm_layers = nn.ModuleList([])
        self.rf_layers = nn.ModuleList([])
        self.sh_encoder = SHEncoder(degree=4)
        self.sh_dim = 16 # degree 4 sh

        self.n_level = 4
        feature_per_level = 2
        self.hash_size = 16
        self.embeddings = nn.ModuleList([nn.Embedding(2**self.hash_size, 2) for i in range(self.n_level)])

        self.lm_layers.append(Siren(dim_in=self.n_level * feature_per_level + self.sh_dim, dim_out=lm_dim, w0=20, is_first=True))

        for _ in range(lm_layer):
            self.lm_layers.append(Siren(dim_in=lm_dim, dim_out=lm_dim))

        self.lm_layers.append(Siren(dim_in=lm_dim, dim_out=dim_hidden))
        
        # rf layers
        self.rf_layers.append(Siren(dim_in=dim_hidden + self.sh_dim, dim_out=rf_dim))
        for _ in range(rf_layer):
            self.rf_layers.append(Siren(dim_in=rf_dim, dim_out=rf_dim))

        self.rf_layers.append(Siren(dim_in=rf_dim, dim_out=3, activation=nn.Sigmoid()))

    def forward(self, uv, n, v):
        '''
        input: 
          uv: B x W x H x 2, 
          n: B x W x H x 3,
          v: B x W x H x 3
        '''
        n_in_sh = self.sh_encoder(n)

        # encode input uv to embedding and concat for inputs
        x_embedded_all = [n_in_sh]
        box_offsets = BOX_OFFSETS
        for i in range(len(uv.shape)-1):
            box_offsets = box_offsets.unsqueeze(0)
        for i in range(self.n_level):
            resolution = 2 ** (i + 9)
            scaled_uv = uv * resolution
            bottom_left_idx = torch.floor(scaled_uv).int()
            corner_indices = bottom_left_idx.unsqueeze(-2) + box_offsets
            hashed_texel_indices = hash(corner_indices, self.hash_size)
            weights = scaled_uv - torch.floor(scaled_uv)
            texel_embedds = self.embeddings[i](hashed_texel_indices)
            embed = texel_embedds[:,:,:,0]*(1-weights[:,:,:,0][:,:,:,None])*(1-weights[:,:,:,1][:,:,:,None]) + \
                    texel_embedds[:,:,:,1]*(1-weights[:,:,:,0][:,:,:,None])*weights[:,:,:,1][:,:,:,None] + \
                    texel_embedds[:,:,:,2]*weights[:,:,:,0][:,:,:,None]*(1-weights[:,:,:,1][:,:,:,None]) + \
                    texel_embedds[:,:,:,3]*weights[:,:,:,0][:,:,:,None]*weights[:,:,:,1][:,:,:,None]
            x_embedded_all.append(embed)
            
        x = torch.cat(x_embedded_all, dim=-1)

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