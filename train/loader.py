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
from pathlib import Path


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

def collect():
    path = Path('dataset_raw/render')
    files = path.glob('**/*.exr').l
    for file in tqdm(path.glob('**/*.exr')):
        print(file.name)


def collect_dataset(img_num=64):
    pn_valid = np.zeros((0, 6), dtype=np.float16)
    v_valid = np.zeros((0, 3), dtype=np.float16)
    color_valid = np.zeros((0, 3), dtype=np.float16)

    path = Path('dataset_raw/render')
    files = [f for f in path.glob('**/*.exr')]
    i = 0
    for file in tqdm(files):
        print(f"load {file.name}")

        pnv_image = load_exr(
            f"dataset_raw/pnv/{file.name}",
            channels=(
                "View Layer.Combined.R",
                "View Layer.Combined.G",
                "View Layer.Combined.B",
                "View Layer.Position.X",
                "View Layer.Position.Y",
                "View Layer.Position.Z",
                "View Layer.Normal.X",
                "View Layer.Normal.Y",
                "View Layer.Normal.Z",
            ),
        )[0:-1:2, 0:-1:2, :]

        pnv_image_concat = pnv_image.reshape([-1, 9])
        normal_length = np.linalg.norm(pnv_image_concat[:, 6:9], axis=1)
        valid_pixel = normal_length >= 0.1

        if i == 0:
            pn0 = pnv_image[:,:, 3:9].astype(np.float16)
        pn_valid = np.concatenate((pn_valid, pnv_image_concat[:, 3:9][valid_pixel]), axis=0)

        if i == 0:
            v0 = pnv_image[:,:, 0:3].astype(np.float16)
        v_valid = np.concatenate((v_valid, pnv_image_concat[:, 0:3][valid_pixel]), axis=0)

        render_img = load_exr(
            str(file),
            channels=(
                'R', 'G', 'B'
            ),
        )[0:-1:2, 0:-1:2, :]
        if i == 0:
            color0 = render_img[:,:, :].astype(np.float16)

        render_img_concat = render_img.reshape([-1, 3])
        color_valid_image = render_img_concat[valid_pixel]
        print(color_valid_image.shape)
        color_valid = np.concatenate(
            (color_valid, color_valid_image), axis=0
        )

        i = i + 1

    np.save(
        f"dataset/render_text",
        {
            "pn0": pn0,
            "v0": v0,
            "color0": color0,
            "pn_valid": pn_valid,
            "v_valid": v_valid,
            "color_valid": color_valid,
        },
        protocol=4
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
    render_img = load_exr(
            f"dataset_raw/render/Camera.exr",
            channels=(
                "R", "G", "B", "A"
            ),
        )
    print(render_img.shape)
    print(render_img.dtype)
    print(render_img[:,:,3])
    img = Image.fromarray((render_img * 255.0).astype(np.uint8))
    img.save('preview.png')

def preview_data():
    dataset = np.load('dataset/render_x.npy', allow_pickle=True)
    pn = dataset.item().get("pn0")
    v = dataset.item().get("v0")
    color = dataset.item().get("color0")
    pn01 = pn
    v01 = v
    img = Image.fromarray((pn01[:,:,0:3] * 255.0).astype(np.uint8))
    img.save('preview1.png')
    img = Image.fromarray((pn01[:,:,3:6] * 255.0).astype(np.uint8))
    img.save('preview2.png')
    img = Image.fromarray((v01[:,:,0:3] * 255.0).astype(np.uint8))
    img.save('preview3.png')
    img = Image.fromarray((color[:,:,0:3] * 255.0).astype(np.uint8))
    img.save('preview4.png')

if __name__ == "__main__":
    # collect()
    collect_dataset()
    # preview_image()
    # preview_data()
