import numpy as np
import torch
import torch.utils.data as Data
import array
import OpenEXR
import Imath
import numpy as np
import math
from tqdm import tqdm


def get_exr_data(path: str):
    file = OpenEXR.InputFile(path)
    dw = file.header()["dataWindow"]
    sz = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    return sz


def load_exr(path: str, channels=("R", "G", "B")):
    file = OpenEXR.InputFile(path)
    dw = file.header()["dataWindow"]
    print(file.header())
    sz = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    chs = [array.array("f", file.channel(Chan, FLOAT)).tolist() for Chan in channels]
    img = (
        np.array(chs)
        .reshape([len(channels), sz[1], sz[0]])
        .transpose((2, 1, 0))
        .clip(0, 1)
    )
    return img


def collect_dataset(img_num=64, down_scale=2, bbx_size=10.0):
    img_data = get_exr_data("dataset_raw/pnc/Camera_00.exr")
    used_size = (
        int(math.floor(img_data[0] / down_scale)),
        int(math.floor(img_data[1] / down_scale)),
    )
    pn = np.zeros((img_num, used_size[0], used_size[1], 6), dtype=np.float16)
    v = np.zeros((img_num, used_size[0], used_size[1], 3), dtype=np.float16)
    uv = np.zeros((img_num, used_size[0], used_size[1], 2), dtype=np.float16)
    color = np.zeros((img_num, used_size[0], used_size[1], 3), dtype=np.float16)
    for i in tqdm(range(img_num)):
        print(f"load {i}")
        pnc_img = load_exr(
            f"dataset_raw/pnc/Camera_{i:02d}.exr",
            channels=(
                "ViewLayer.Position.X",
                "ViewLayer.Position.Y",
                "ViewLayer.Position.Z",
                "ViewLayer.Normal.X",
                "ViewLayer.Normal.Y",
                "ViewLayer.Normal.Z",
                "ViewLayer.Combined.R",
                "ViewLayer.Combined.G",
                "ViewLayer.Combined.B",
            ),
        )
        pn[i, :, :, 0:3] = pnc_img[0:-1:down_scale, 0:-1:down_scale, 0:3] / bbx_size
        pn[i, :, :, 3:6] = pnc_img[0:-1:down_scale, 0:-1:down_scale, 3:6]
        color[i, :, :, :] = pnc_img[0:-1:down_scale, 0:-1:down_scale, 6:9]

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
        v[i, :, :, :] = vuv_img[0:-1:down_scale, 0:-1:down_scale, 0:3]
        uv[i, :, :, :] = vuv_img[0:-1:down_scale, 0:-1:down_scale, 3:5]

    np.save(
        f"dataset/render_x-{down_scale}", {"pn": pn, "v": v, "uv": uv, "color": color}
    )


def prepare_dataloader(path="dataset/render_x-2.npy", batch_size=10):
    dataset = np.load(path, allow_pickle=True)
    pn = dataset.item().get("pn")
    v = dataset.item().get("v")
    color = dataset.item().get("color")

    train_inputs_pn = pn
    train_inputs_v = v
    test_inputs_pn = pn[0 : pn.shape[0] : 8, :, :, :]
    test_inputs_v = v[0 : v.shape[0] : 8, :, :, :]

    train_output_color = color[:, :, :, :]
    test_output_color = color[0 : color.shape[0] : 8, :, :, :]

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
        pn.shape[1:3],
    )


if __name__ == "__main__":
    collect_dataset()
