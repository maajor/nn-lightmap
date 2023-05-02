import torch
import numpy as np
import matplotlib.pyplot as plt
from model import SirenGINet
import time
from PIL import Image


def load_and_vis(
    model, name, epoch=None, writer=None, dataset_path="dataset/render_monkey_128.npz"
):
    device = torch.device("cuda:0")

    dataset = np.load(dataset_path, allow_pickle=True)

    p = dataset["pn0"][0:-1:2, 0:-1:2, 0:3]
    n = dataset["pn0"][0:-1:2, 0:-1:2, 3:6]
    v = dataset["v0"][0:-1:2, 0:-1:2, :]
    input_n = torch.from_numpy(n).float()
    input_n = input_n.to(device)
    input_v = torch.from_numpy(v).float()
    input_v = input_v.to(device)
    input_p = torch.from_numpy(p).float()
    input_p = input_p.to(device)
    model.eval()
    with torch.no_grad():
        tim = time.time()
        pred_output = model(input_p, input_n, input_v)
        diff = time.time() - tim
        print(diff * 1000)
    img = pred_output.cpu().detach().numpy()[:,:,:]

    # img_pil = Image.fromarray((img * 255.0).astype(np.uint8))
    # img_pil.save(f"model/{name}.png")
    if writer is not None:
        writer.add_images("output", img, global_step=epoch, dataformats="HWC")


if __name__ == "__main__":
    model = SirenGINet(256, 5, 8, 32, 2)
    model.load_state_dict(torch.load("model/model_siren_256x5x32x64x38x32x2.pth"))

    device = torch.device("cuda:0")
    model = model.to(device)
    load_and_vis(model, "model_siren")
