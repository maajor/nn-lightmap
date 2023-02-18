import torch
import OpenEXR
import Imath
import math
import numpy as np
import array
from PIL import Image
from model import SirenGINet

model = SirenGINet(256, 5, 32, 32, 3)
model.load_state_dict(torch.load("model/model_siren_256x5x32x32x3_2400.pth")) 

def load_exr(path: str, channels=("R", "G", "B")):
    file = OpenEXR.InputFile(path)
    dw = file.header()["dataWindow"]
    sz = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    chs = [array.array("f", file.channel(Chan, FLOAT)).tolist() for Chan in channels]
    img = (
        np.array(chs)
        .reshape([len(channels), sz[1], sz[0]])
        .transpose((2, 1, 0))
    )
    return img

def get_uv():
    uv = load_exr('lightmap/vuv0.exr', ('ViewLayer.UV.U', 'ViewLayer.UV.V'))
    return uv

def get_groundtruth_pn():
    dataset = np.load("dataset/render_2k.npy", allow_pickle=True)
    pn = dataset.item().get("pn0")[:, :, :] 
    img_pil = Image.fromarray((pn[:,:,0:3] * 255.0).astype(np.uint8))
    img_pil.save(f"groundtruth_p.png")
    img_pil = Image.fromarray((pn[:,:,3:6] * 255.0).astype(np.uint8))
    img_pil.save(f"groundtruth_n.png")
    return pn

def get_groundtruth_v():
    dataset = np.load("dataset/render_2k.npy", allow_pickle=True)
    v = dataset.item().get("v0")[:, :, :] 
    return v

def bilinear_sample_texture(u0, v0, u1, v1, ru, rv, texture):
    return texture[u0, v0, :] * (1-ru) * (1-rv) + texture[u1, v0, :] * (ru) * (1-rv) + texture[u0, v1, :] * (1-ru) * (rv) + texture[u1, v1, :] * (ru) * (rv)


def get_lightmap_pn():
    uv = get_uv()
    position = load_exr('lightmap/position.exr') * 2.0 - 1.0
    normal = load_exr('lightmap/normal.exr') * 2.0 - 1.0
    lmw, lmh, _ = position.shape
    w, h, c = uv.shape
    npview = np.zeros((w, h, 6))
    for i in range(w):
        for j in range(h):
            uvpix, _ = np.modf(uv[i, j])
            u_0 = min(math.floor(uvpix[0] * lmw), lmw-1)
            v_0 = min(math.floor((1 - uvpix[1])*lmh), lmh-1)
            u_1 = min(u_0+1, lmw-1)
            v_1 = min(v_0+1, lmh-1)
            r_u = uvpix[0] * lmw - u_0
            r_v = (1 - uvpix[1])*lmh - v_0
            npview[i,j,0:3]= bilinear_sample_texture(u_0, v_0, u_1, v_1, r_u, r_v, position)
            npview[i,j,3:6]= bilinear_sample_texture(u_0, v_0, u_1, v_1, r_u, r_v, normal)
            # npview[i,j,0:3]= position[u_0, v_0, :]
            # npview[i,j,3:6]= normal[u_0, v_0, :]
    img_pil = Image.fromarray((npview[:,:,0:3] * 255.0).astype(np.uint8))
    img_pil.save(f"lightmap_p_bi.png")
    img_pil = Image.fromarray((npview[:,:,3:6] * 255.0).astype(np.uint8))
    img_pil.save(f"lightmap_n_bi.png")
    return npview


def predict_with_gt_pn():
    pn = get_groundtruth_pn()
    v = get_groundtruth_v()

    model.eval()
    with torch.no_grad():
        pn = torch.from_numpy(pn).float()
        v = torch.from_numpy(v).float()
        pred_output = model(pn, v)
        pred_output = pred_output.numpy()
        img_pil = Image.fromarray((pred_output[:,:,0:3] * 255.0).astype(np.uint8))
        img_pil.save(f"predict_with_gt_pn.png")


def predict_with_lightmap_pn():
    pn = get_lightmap_pn()
    v = get_groundtruth_v()

    model.eval()
    with torch.no_grad():
        pn = torch.from_numpy(pn).float()
        v = torch.from_numpy(v).float()
        pred_output = model(pn, v)
        pred_output = pred_output.numpy()
        img_pil = Image.fromarray((pred_output[:,:,0:3] * 255.0).astype(np.uint8))
        img_pil.save(f"predict_with_lightmap_pn.png")

def get_lightmap():
    pn = get_lightmap_pn()

    model.eval()
    with torch.no_grad():
        pn = torch.from_numpy(pn).float()
        lm = model.bake_lightmap(pn)
        lm = lm.numpy()
    return lm

def bake_lightmap():
    lm = get_lightmap()
    w,h,c = lm.shape
    for i in range(0, c, 4):
        channels = lm[:,:,i:i+4]
        min = np.min(channels)
        max = np.max(channels)
        print(f'channel {i}-{i+4} has min {min} and max {max}')
        channels = (channels + min) / (max - min)
        img_pil = Image.fromarray((channels * 255.0).astype(np.uint8))
        img_pil.save(f"lightmap_{i/4}.png")



def inference_with_lightmap():
    lightmap = get_lightmap()
    view = get_groundtruth_v()

    model.eval()
    with torch.no_grad():
        lightmap = torch.from_numpy(lightmap).float()
        v = torch.from_numpy(view).float()
        render = model.inference_with_lightmap(lightmap, v)
        render = render.numpy()

        img_pil = Image.fromarray((render * 255.0).astype(np.uint8))
        img_pil.save(f"render_with_lightmap.png")

if __name__ == '__main__':
    bake_lightmap()
    # inference_with_lightmap()