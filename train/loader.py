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
import json

DATASET_NAME = "dataset/render_monkey_128"

def get_exr_data(path: str):
    '''
    Get the data from an EXR image.
    '''
    file = OpenEXR.InputFile(path)
    dw = file.header()["dataWindow"]
    sz = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    return sz


def load_exr(path: str, channels=("R", "G", "B")):
    '''Load an EXR image from path as a numpy array. 
    '''
    file = OpenEXR.InputFile(path)
    dw = file.header()["dataWindow"]
    # print(file.header())
    sz = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    chs = [array.array("f", file.channel(Chan, FLOAT)).tolist() for Chan in channels]
    img = np.array(chs).reshape([len(channels), sz[1], sz[0]]).transpose((2, 1, 0))
    return img


def collect():
    '''
    Collect the dataset from the render folder. Just for debug
    '''
    path = Path("dataset_raw/render")
    files = path.glob("**/*.exr")
    for file in tqdm(path.glob("**/*.png")):
        print(file.stem)


def collect_dataset():
    '''
    Collect the dataset from the render folder
    '''
    # position and normal information, only valid pixels
    pn_valid = np.zeros((0, 6), dtype=np.float16)
    # view direction information, only valid pixels
    v_valid = np.zeros((0, 3), dtype=np.float16)
    # color information, only valid pixels
    color_valid = np.zeros((0, 3), dtype=np.float16)

    path = Path("dataset_raw/render")
    files = [f for f in path.glob("**/*.png")]
    i = 0
    with open('dataset_raw/cam_pos.json', 'r') as f:
        cam_pos = json.load(f)
    for file in tqdm(files):
        print(f"load {file.stem}")
        camera_position = np.array(cam_pos[file.stem])

        render_image = Image.open(f'dataset_raw/render/{file.stem}.png')
        render_image = np.array(render_image) / 255.0
        render_image_concat = render_image.reshape([-1, 4])
        # if render channel's alpha is larger than 0, it's a valid pixel
        valid_pixel = render_image_concat[:,3] >= (1 - 1e-6)

        if i == 0:
            color0 = render_image[:, :, 0:3].astype(np.float16)
        color_valid = np.concatenate((color_valid, render_image_concat[:, 0:3][valid_pixel]), axis=0)

        # pn_image is resolution x resolution x 6 shape
        pn_image = load_exr(
            f"dataset_raw/pn/{file.stem}.exr",
            channels=(
                "ViewLayer.Position.X",
                "ViewLayer.Position.Y",
                "ViewLayer.Position.Z",
                "ViewLayer.Normal.X",
                "ViewLayer.Normal.Y",
                "ViewLayer.Normal.Z",
            ),
        )
        pn_image = np.swapaxes(pn_image,0,1)

        pn_image_concat = pn_image.reshape([-1, 6])
        # valid pixels are the same as render image
        valid_pn_image = pn_image_concat[valid_pixel]
        normal_norm = np.linalg.norm(valid_pn_image[:,3:6], axis=-1)
        normal_norm = np.maximum(normal_norm, 0.1)
        # normalize normal
        valid_pn_image[:, 3:6] = valid_pn_image[:, 3:6] / normal_norm[:,None]

        if i == 0:
            # first image for test
            pn0 = pn_image.astype(np.float16)
        pn_valid = np.concatenate(
            (pn_valid, valid_pn_image), axis=0
        )

        if i == 0:
            # first image for test
            v0 = pn_image[:, :, 0:3] - camera_position[np.newaxis, np.newaxis, :]
            v0 = v0 / np.linalg.norm(v0, axis=-1)[:, :, np.newaxis]
            v0 = v0.astype(np.float16)

        # view direction is calculated from "position - camera position"
        # and then we normalize it
        view_should_be = valid_pn_image[:, 0:3] - camera_position[np.newaxis, :]
        view_should_be = view_should_be / np.linalg.norm(view_should_be, axis=-1)[:,None]

        v_valid = np.concatenate(
            (v_valid, view_should_be), axis=0
        )

        i = i + 1

    np.savez_compressed(
        DATASET_NAME,
        pn0=pn0,
        v0=v0,
        color0=color0,
        pn_valid=pn_valid,
        v_valid=v_valid,
        color_valid=color_valid
    )


def prepare_dataloader(path, batch_size=100):
    dataset = np.load(path, allow_pickle=True)
    pn = dataset["pn_valid"]
    v = dataset["v_valid"]
    color = dataset["color_valid"]

    shape_l = pn.shape[0]
    # reshape n x c input data to w x h x c data,
    w, h = (128, 128)
    reshape_nums = int(math.floor(shape_l / (w * h)))
    reshape_all_size = reshape_nums * w * h

    train_inputs_p = pn[0:reshape_all_size, 0:3].reshape(-1, w, h, 3)
    train_inputs_n = pn[0:reshape_all_size, 3:6].reshape(-1, w, h, 3)
    train_inputs_v = v[0:reshape_all_size, :].reshape(-1, w, h, 3)
    test_inputs_p = train_inputs_p[0 : train_inputs_p.shape[0] : 8, :, :]
    test_inputs_n = train_inputs_n[0 : train_inputs_n.shape[0] : 8, :, :]
    test_inputs_v = train_inputs_v[0 : train_inputs_v.shape[0] : 8, :, :]

    train_output_color = color[0:reshape_all_size, :].reshape(-1, w, h, 3)
    test_output_color = train_output_color[0 : train_output_color.shape[0] : 8, :, :] 

    def loader(p, n, v, color, batch_size):
        p = torch.from_numpy(p).float()
        n = torch.from_numpy(n).float()
        v = torch.from_numpy(v).float()
        color = torch.from_numpy(color).float()

        data_set = Data.TensorDataset(p, n, v, color)

        loader = Data.DataLoader(
            dataset=data_set, batch_size=batch_size, shuffle=True, num_workers=1,
        )
        return loader

    return (
        loader(train_inputs_p, train_inputs_n, train_inputs_v, train_output_color, batch_size),
        loader(test_inputs_p, test_inputs_n, test_inputs_v, test_output_color, batch_size),
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
    '''
    preview npz dataset's first image for test
    '''
    dataset = np.load(f"{DATASET_NAME}.npz", allow_pickle=True)
    pn01 = dataset['pn0'] * 0.5 + 0.5
    v01 = dataset['v0'] * 0.5 + 0.5
    color = dataset['color0']
    print(f"view range {np.min(dataset['v_valid'])} ~ {np.max(dataset['v_valid'])}")
    print(np.max(dataset['color0']))
    print(np.max(dataset['color_valid']))
    print(dataset['pn_valid'].shape)
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
    collect_dataset()
    # preview_image()
    preview_data()
