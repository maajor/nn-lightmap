import torch
import OpenEXR
import Imath
import math
import numpy as np
import array
from PIL import Image
from model import SirenGINet

model = SirenGINet(256, 5, 16, 64, 2)
model.load_state_dict(torch.load("model/model_siren_256x5x16x64x2_300.pth"))


def load_exr(path: str, channels=("R", "G", "B")):
    file = OpenEXR.InputFile(path)
    dw = file.header()["dataWindow"]
    sz = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    chs = [array.array("f", file.channel(Chan, FLOAT)).tolist() for Chan in channels]
    img = np.array(chs).reshape([len(channels), sz[1], sz[0]]).transpose((2, 1, 0))
    return img


def get_uv():
    uv = load_exr("dataset_raw/uv/Camera.002.exr", ("View Layer.UV.U", "View Layer.UV.V"))
    uv = np.swapaxes(uv,0,1)
    w, h, c = uv.shape
    ones = np.ones((w,h,2))
    uvimg = np.concatenate([uv, ones], -1)
    img_uv = Image.fromarray((uvimg * 255.0).astype(np.uint8))
    img_uv.save(f"image_uv.png")
    return uv[0:-1:4, 0:-1:4, :]


def get_groundtruth_pn():
    dataset = np.load("dataset/render_text_4k.npz", allow_pickle=True)
    pn = dataset['pn0'][0:-1:4, 0:-1:4, :]
    normalized = pn * 0.5 + 0.5
    img_pil = Image.fromarray((normalized[:, :, 0:3] * 255.0).astype(np.uint8))
    img_pil.save(f"groundtruth_p.png")
    img_pil = Image.fromarray((normalized[:, :, 3:6] * 255.0).astype(np.uint8))
    img_pil.save(f"groundtruth_n.png")
    return pn


def get_groundtruth_v():
    dataset = np.load("dataset/render_text_4k.npz", allow_pickle=True)
    v = dataset["v0"][0:-1:4, 0:-1:4, :]
    print(np.max(v))
    print(np.min(v))
    img_pil = Image.fromarray((v[:, :, 0:3] * 255.0).astype(np.uint8))
    img_pil.save(f"groundtruth_v.png")
    return v


def bilinear_sample_texture(u0, v0, u1, v1, ru, rv, texture):
    return (
        texture[u0, v0, :] * (1 - ru) * (1 - rv)
        + texture[u1, v0, :] * (ru) * (1 - rv)
        + texture[u0, v1, :] * (1 - ru) * (rv)
        + texture[u1, v1, :] * (ru) * (rv)
    )


def get_lightmap_pn():
    position = load_exr("lightmap/text/position.exr") * 2.0 - 1.0
    normal = load_exr("lightmap/text/normal.exr") * 2.0 - 1.0
    position = np.swapaxes(position,0,1)
    normal = np.swapaxes(normal,0,1)
    return np.concatenate((position, normal), -1)


def get_lightmap_pn_view():
    uv = get_uv()
    position = load_exr("lightmap/text/position.exr") * 2.0 - 1.0
    normal = load_exr("lightmap/text/normal.exr") * 2.0 - 1.0
    position = np.swapaxes(position,0,1)
    normal = np.swapaxes(normal,0,1)
    lmw, lmh, _ = position.shape
    w, h, c = uv.shape
    npview = np.zeros((w, h, 6))
    print('start get lightmap pn view')
    for i in range(w):
        if i % 100 == 0:
            print(i)    
        for j in range(h):
            uvpix, _ = np.modf(uv[i, j])
            u_0 = min(math.floor((1-uvpix[1]) * lmw), lmw - 1)
            v_0 = min(math.floor((uvpix[0]) * lmh), lmh - 1)
            u_1 = min(u_0 + 1, lmw - 1)
            v_1 = min(v_0 + 1, lmh - 1)
            r_u = (1-uvpix[1]) * lmw - u_0
            r_v = (uvpix[0]) * lmh - v_0
            npview[i, j, 0:3] = bilinear_sample_texture(
                u_0, v_0, u_1, v_1, r_u, r_v, position
            )
            npview[i, j, 3:6] = bilinear_sample_texture(
                u_0, v_0, u_1, v_1, r_u, r_v, normal
            )
            # npview[i,j,0:3]= position[u_0, v_0, :]
            # npview[i,j,3:6]= normal[u_0, v_0, :]
    # img_pil = Image.fromarray((npview[:,:,0:3] * 255.0).astype(np.uint8))
    # img_pil.save(f"lightmap_p_bi.png")
    # img_pil = Image.fromarray((npview[:,:,3:6] * 255.0).astype(np.uint8))
    # img_pil.save(f"lightmap_n_bi.png")
    return npview


def predict_with_gt_pn():
    pn = get_groundtruth_pn()
    v = get_groundtruth_v()

    model.eval()
    print('predict_with_gt_pn')
    with torch.no_grad():
        pn = torch.from_numpy(pn).float()
        v = torch.from_numpy(v).float()
        pred_output = model(pn, v)
        pred_output = pred_output.numpy()
        img_pil = Image.fromarray((pred_output[:, :, 0:3] * 255.0).astype(np.uint8))
        img_pil.save(f"predict_with_gt_pn.png")


def predict_with_lightmap_pn():
    pn = get_lightmap_pn_view()
    v = get_groundtruth_v()

    model.eval()
    print('predict_with_lightmap_pn')
    with torch.no_grad():
        pn = torch.from_numpy(pn).float()
        v = torch.from_numpy(v).float()
        pred_output = model(pn, v)
        pred_output = pred_output.numpy()
        img_pil = Image.fromarray((pred_output[:, :, 0:3] * 255.0).astype(np.uint8))
        img_pil.save(f"predict_with_lightmap_pn.png")


def get_lightmap_from_view():
    pn = get_lightmap_pn_view()

    model.eval()
    with torch.no_grad():
        pn = torch.from_numpy(pn).float()
        lm = model.bake_lightmap(pn[...,0:3], pn[...,3:6])
        lm = lm.numpy()
    return lm


def get_lightmap_from_uv():
    pn = get_lightmap_pn()

    model.eval()
    with torch.no_grad():
        pn = torch.from_numpy(pn).float()
        lm = model.bake_lightmap(pn[...,0:3], pn[...,3:6])
        lm = lm.numpy()
    return lm


def bake_lightmap():
    lm = get_lightmap_from_uv()
    w, h, c = lm.shape
    for i in range(0, c, 4):
        channels = lm[:, :, i : i + 4]
        # channels in range -1 ~ 1
        channels = channels * 0.5 + 0.5
        img_pil = Image.fromarray((channels * 255.0).astype(np.uint8))
        img_pil.save(f"lightmap_{i/4}.webp", quality=80)


def load_lightmap():
    channs = []
    for i in range(4):
        img = Image.open(f'lightmap_{i}.0.webp')
        channs.append(np.array(img))
    return np.concatenate(channs, -1)

def assert_lms():
    load_lms = load_lightmap()
    lms_from_uv = get_lightmap_from_uv() * 0.5 + 0.5
    lms_from_uv = (lms_from_uv * 255.0).astype(np.uint8)
    print(np.max(load_lms - lms_from_uv))


def inference_with_lightmap():
    lightmap = get_lightmap_from_view()
    view = get_groundtruth_v()

    model.eval()
    with torch.no_grad():
        lightmap = torch.from_numpy(lightmap).float()
        v = torch.from_numpy(view).float()
        render = model.inference_with_lightmap(lightmap, v)
        render = render.numpy()

        img_pil = Image.fromarray((render * 255.0).astype(np.uint8))
        img_pil.save(f"render_with_lightmap.png")

def debug_lightmap():
    lm = np.ones((32))
    v = np.array([0.5, 0.5, 0.5])

    model.eval()
    with torch.no_grad():
        lmt = torch.from_numpy(lm).float()
        vt = torch.from_numpy(v).float()
        pred_output = model.inference_with_lightmap(lmt, vt)
        # print(pred_output)
        print((pred_output)*255)
        #w = model.rf_layers[0].weight.detach().numpy()
        #b = model.rf_layers[0].bias.detach().numpy()
        # a = model.vup_layers.activation.w0
        #print(w)
        #print(b)
        # print(np.matmul(w, v))
        # print(np.sin((np.matmul(w, v) + b)*a))


def predict_from_datasource(name: str):
    pn_image = load_exr(
        f"dataset_raw/pn/{name}.exr",
        channels=(
            "View Layer.Position.X",
            "View Layer.Position.Y",
            "View Layer.Position.Z",
            "View Layer.Normal.X",
            "View Layer.Normal.Y",
            "View Layer.Normal.Z",
        ),
    )
    pn_image = np.swapaxes(pn_image,0,1)[0:-1:2, 0:-1:2, :]

    view_img = load_exr(str(f"dataset_raw/viewold/{name}.exr"), channels=("R", "G", "B"))
    view_img = np.swapaxes(view_img,0,1)[0:-1:2, 0:-1:2, :]

    model.eval()
    with torch.no_grad():
        pn = torch.from_numpy(pn_image).float()
        v = torch.from_numpy(view_img).float()
        render = model(pn, v)
        render = render.numpy()

        img_pil = Image.fromarray((render * 255.0).astype(np.uint8))
        img_pil.save(f"render_{name}.png")




if __name__ == "__main__":
    bake_lightmap()
    # predict_with_gt_pn()
    # predict_with_lightmap_pn()
    # inference_with_lightmap()
    # get_uv()
    # predict_with_gt_pn()
    # debug_lightmap()
    # predict_from_datasource('Camera.028')
