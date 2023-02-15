import torch
import numpy as np
import matplotlib.pyplot as plt
from model import SirenGINet
import time
from PIL import Image


def load_and_vis(
    model, name, epoch, writer, dataset_path="dataset/render_2k.npy"
):
    device = torch.device("cuda:0")

    dataset = np.load(dataset_path, allow_pickle=True)
    pn = dataset.item().get("pn0")
    v = dataset.item().get("v0")
    input_pn = torch.from_numpy(pn).float()
    input_pn = input_pn.to(device)
    input_v = torch.from_numpy(v).float()
    input_v = input_v.to(device)
    model.eval()
    with torch.no_grad():
        tim = time.time()
        pred_output = model(input_pn, input_v)
        diff = time.time() - tim
        print(diff * 1000)
    img = pred_output.cpu().detach().numpy()

    # img_pil = Image.fromarray((img * 255.0).astype(np.uint8))
    # img_pil.save(f"model/{name}.png")
    writer.add_images('output', img, global_step=epoch, dataformats='HWC')


if __name__ == "__main__":
    model = SirenGINet()

    device = torch.device("cuda:0")
    model.load_state_dict(torch.load("model/model_siren.pth"))
    model = model.to(device)
    load_and_vis(model, "model_siren")
