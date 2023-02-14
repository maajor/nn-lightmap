import torch
import numpy as np
import matplotlib.pyplot as plt
from model import SirenGINet
import time


def load_and_vis(
    model, name, dataset_path="dataset/render_x.npy"
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

    fig = plt.figure(figsize=(10, 7))
    # fig.add_subplot(1, 2, 1)
    plt.imshow(img[:, :, 0:3])
    # fig.add_subplot(1, 2, 2)
    # plt.imshow(color[image_id, :, :, 0:3])
    # plt.waitforbuttonpress()
    plt.savefig(f"{name}.png")


if __name__ == "__main__":
    model = SirenGINet()

    device = torch.device("cuda:0")
    model.load_state_dict(torch.load("model/model_siren.pth"))
    model = model.to(device)
    load_and_vis(model, "model_siren")
