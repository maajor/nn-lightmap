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
    img = np.array(chs).reshape([len(channels), sz[1], sz[0]]).transpose((2, 1, 0))
    return img


def collect():
    path = Path("dataset_raw/render")
    files = path.glob("**/*.exr")
    for file in tqdm(path.glob("**/*.png")):
        print(file.stem)


def collect_dataset():
    pn_valid = np.zeros((0, 6), dtype=np.float16)
    v_valid = np.zeros((0, 3), dtype=np.float16)
    color_valid = np.zeros((0, 3), dtype=np.float16)

    path = Path("dataset_raw/render")
    files = [f for f in path.glob("**/*.png")]
    i = 0
    for file in tqdm(files):
        print(f"load {file.stem}")

        render_image = Image.open(f'dataset_raw/render/{file.stem}.png')
        render_image = np.array(render_image) / 255.0
        render_image_concat = render_image.reshape([-1, 4])
        valid_pixel = render_image_concat[:,3] >= 0.1

        if i == 0:
            color0 = render_image[:, :, 0:3].astype(np.float16)
        color_valid = np.concatenate((color_valid, render_image_concat[:, 0:3][valid_pixel]), axis=0)
        print(f"max color value {np.max(color_valid)}")

        pn_image = load_exr(
            f"dataset_raw/pn/{file.stem}.exr",
            channels=(
                "View Layer.Position.X",
                "View Layer.Position.Y",
                "View Layer.Position.Z",
                "View Layer.Normal.X",
                "View Layer.Normal.Y",
                "View Layer.Normal.Z",
            ),
        )
        pn_image = np.swapaxes(pn_image,0,1)

        pn_image_concat = pn_image.reshape([-1, 6])

        if i == 0:
            pn0 = pn_image.astype(np.float16)
        pn_valid = np.concatenate(
            (pn_valid, pn_image_concat[valid_pixel]), axis=0
        )

        view_img = load_exr(str(f"dataset_raw/view/{file.stem}.exr"), channels=("R", "G", "B"))
        view_img = np.swapaxes(view_img,0,1)
        if i == 0:
            v0 = view_img.astype(np.float16)

        view_img_concat = view_img.reshape([-1, 3])
        view_valid_image = view_img_concat[valid_pixel]
        print(view_valid_image.shape)

        v_valid = np.concatenate(
            (v_valid, view_valid_image), axis=0
        )

        i = i + 1

    np.savez_compressed(
        f"dataset/render_text_4k",
        pn0=pn0,
        v0=v0,
        color0=color0,
        pn_valid=pn_valid,
        v_valid=v_valid,
        color_valid=color_valid
    )


def prepare_dataloader(path="dataset/render_text_4k.npz", batch_size=100):
    dataset = np.load(path, allow_pickle=True)
    pn = dataset["pn_valid"]
    v = dataset["v_valid"]
    color = dataset["color_valid"].clip(0,1)

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
        (w, h, 3),
    )


def preview_image():
    render = Image.open('dataset_raw/render/Camera.png')
    data = np.array(render) / 255.0
    print(np.max(data))
    print(data.shape)
    pn_image = load_exr(
        f"dataset_raw/pn/Camera.exr", channels=("View Layer.Normal.X","View Layer.Normal.Y","View Layer.Normal.Z"),
    ) * 0.5 + 0.5
    pn_image = np.swapaxes(pn_image,0,1)
    # pn_image = np.flip(pn_image, 1)
    # pn_image = np.flip(pn_image, 0)
    print(pn_image.shape)
    print(pn_image.dtype)
    img = Image.fromarray((pn_image * 255.0).astype(np.uint8))
    img.save("preview_n.png")
    img = Image.fromarray((data).astype(np.uint8))
    img.save("preview_render.png")


def preview_data():
    dataset = np.load("dataset/render_text_4k.npz", allow_pickle=True)
    pn01 = dataset['pn0'] * 0.5 + 0.5
    v01 = dataset['v0'] * 0.5 + 0.5
    color = dataset['color0']
    print(np.max(dataset['color0']))
    print(np.max(dataset['color_valid']))
    img = Image.fromarray((pn01[:, :, 0:3] * 255.0).astype(np.uint8))
    img.save("preview1.png")
    img = Image.fromarray((pn01[:, :, 3:6] * 255.0).astype(np.uint8))
    img.save("preview2.png")
    img = Image.fromarray((v01[:, :, 0:3] * 255.0).astype(np.uint8))
    img.save("preview3.png")
    img = Image.fromarray((color[:, :, 0:3] * 255.0).astype(np.uint8))
    img.save("preview4.png")


if __name__ == "__main__":
    # collect()
    # collect_dataset()
    # preview_image()
    preview_data()
