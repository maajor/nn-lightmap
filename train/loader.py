import numpy as np
import torch
import torch.utils.data as Data
import array
import OpenEXR
import Imath
import numpy as np
import math
from tqdm import tqdm
from PIL import Image


def get_exr_data(path: str):
    file = OpenEXR.InputFile(path)
    dw = file.header()["dataWindow"]
    sz = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    return sz


def load_exr(path: str, channels=("R", "G", "B")):
    file = OpenEXR.InputFile(path)
    dw = file.header()["dataWindow"]
    # print(file.header())
    sz = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    chs = [array.array("f", file.channel(Chan, FLOAT)).tolist() for Chan in channels]
    img = (
        np.array(chs)
        .reshape([len(channels), sz[1], sz[0]])
        .transpose((2, 1, 0))
    )
    return img


def collect_dataset(img_num=64):
    img_data = get_exr_data("dataset_raw/render/Camera_00.exr")
    used_size = (
        int(math.floor(img_data[0])),
        int(math.floor(img_data[1])),
    )
    pn = np.zeros((img_num, used_size[0], used_size[1], 6), dtype=np.float16)
    v = np.zeros((img_num, used_size[0], used_size[1], 3), dtype=np.float16)
    uv = np.zeros((img_num, used_size[0], used_size[1], 2), dtype=np.float16)
    color = np.zeros((img_num, used_size[0], used_size[1], 3), dtype=np.float16)

    pn_valid = np.zeros((0, 6), dtype=np.float16)
    v_valid = np.zeros((0, 3), dtype=np.float16)
    color_valid = np.zeros((0, 3), dtype=np.float16)

    for i in tqdm(range(img_num)):
        print(f"load {i}")
        render_img = load_exr(
            f"dataset_raw/render/Camera_{i:02d}.exr",
            channels=(
                "ViewLayer.Combined.R",
                "ViewLayer.Combined.G",
                "ViewLayer.Combined.B",
                "ViewLayer.Normal.X",
                "ViewLayer.Normal.Y",
                "ViewLayer.Normal.Z",
            ),
        )
        color[i, :, :, :] = render_img[:, :, 0:3]

        render_img_concat = render_img.reshape([-1, 6])
        normal_length = np.linalg.norm(render_img_concat[:, 3:6], axis=1)
        valid_pixel = normal_length >= 0.1
        color_valid = np.concatenate(
            (color_valid, render_img_concat[:, 0:3][valid_pixel]), axis=0
        )

        position_img = load_exr(
            f"dataset_raw/position/Camera_{i:02d}.exr",
            channels=(
                "ViewLayer.Combined.R",
                "ViewLayer.Combined.G",
                "ViewLayer.Combined.B",
            ),
        )

        position_img_concat = position_img.reshape([-1, 3]) * 2.0 - 1.0
        pn[i, :, :, 0:3] = position_img[:, :, :] * 2.0 - 1.0

        normal_img = load_exr(
            f"dataset_raw/normal/Camera_{i:02d}.exr",
            channels=(
                "ViewLayer.Combined.R",
                "ViewLayer.Combined.G",
                "ViewLayer.Combined.B",
            ),
        )
        normal_img_concat = normal_img.reshape([-1, 3]) * 2.0 - 1.0
        pn[i, :, :, 3:6] = normal_img[:, :, :] * 2.0 - 1.0

        pn_concat = np.concatenate(
            (position_img_concat[valid_pixel], normal_img_concat[valid_pixel]), axis=1
        )
        pn_valid = np.concatenate((pn_valid, pn_concat), axis=0)

        vuv_img = load_exr(
            f"dataset_raw/vuv/Camera_{i:02d}.exr",
            channels=(
                "ViewLayer.Combined.R",
                "ViewLayer.Combined.G",
                "ViewLayer.Combined.B",
                "ViewLayer.UV.U",
                "ViewLayer.UV.V",
            ),
        )
        vuv_img_concat = vuv_img.reshape([-1, 5]) * 2.0 - 1.0
        v[i, :, :, :] = vuv_img[:, :, 0:3] * 2.0 - 1.0
        uv[i, :, :, :] = vuv_img[:, :, 3:5]
        v_valid = np.concatenate((v_valid, vuv_img_concat[:, 0:3][valid_pixel]), axis=0)

    np.save(
        f"dataset/render_2k",
        {
            "pn0": pn[0, :, :, :],
            "v0": v[0, :, :, :],
            # "uv0": uv[0, :, :, :],
            "color0": color[0, :, :, :],
            "pn_valid": pn_valid,
            "v_valid": v_valid,
            "color_valid": color_valid,
        },
    )


def prepare_dataloader(path="dataset/render_2k.npy", batch_size=100):
    dataset = np.load(path, allow_pickle=True)
    pn = dataset.item().get("pn_valid")
    v = dataset.item().get("v_valid")
    color = dataset.item().get("color_valid")

    shape_l = pn.shape[0]
    w, h = (512, 512)
    reshape_nums = int(math.floor(shape_l / (w * h)))
    reshape_all_size = reshape_nums * w * h

    train_inputs_pn = pn[0:reshape_all_size, :].reshape(-1, w, h, 6)
    train_inputs_v = v[0:reshape_all_size, :].reshape(-1, w, h, 3)
    test_inputs_pn = train_inputs_pn[0 : train_inputs_pn.shape[0] : 8, :, :, :]
    test_inputs_v = train_inputs_v[0 : train_inputs_v.shape[0] : 8, :, :, :]

    train_output_color = color[0:reshape_all_size, :].reshape(-1, w, h, 3)
    test_output_color = train_output_color[0 : train_output_color.shape[0] : 8, :, :, :]

    def loader(pn, v, color, batch_size):
        pn = torch.from_numpy(pn).float()
        v = torch.from_numpy(v).float()
        color = torch.from_numpy(color).float()

        data_set = Data.TensorDataset(pn, v, color)

        loader = Data.DataLoader(
            dataset=data_set, batch_size=batch_size, shuffle=True, num_workers=1,
        )
        return loader

    return (
        loader(train_inputs_pn, train_inputs_v, train_output_color, batch_size),
        loader(test_inputs_pn, test_inputs_v, test_output_color, batch_size),
        (w, h),
    )


def preview_image():
    position_img = load_exr(
            f"dataset_raw/normal/Camera_00.exr",
            channels=(
                "ViewLayer.Combined.R",
                "ViewLayer.Combined.G",
                "ViewLayer.Combined.B",
            ),
        )
    print(position_img.shape)
    print(position_img.dtype)
    img = Image.fromarray((position_img * 255.0).astype(np.uint8))
    img.save('preview.png')

if __name__ == "__main__":
    collect_dataset()
    # preview_image()
