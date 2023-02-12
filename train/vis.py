import torch
import numpy as np
import matplotlib.pyplot as plt
from model import SirenGINet
import time

def load_and_vis(model_checkpoint_path, dataset_path="dataset/render_x-2.npy", image_id=0):
    model = SirenGINet()

    device = torch.device("cuda:0")
    model.load_state_dict(torch.load(model_checkpoint_path))
    model = model.to(device)

    dataset = np.load(dataset_path, allow_pickle=True)
    pn = dataset.item().get("pn")
    v = dataset.item().get("v")
    color = dataset.item().get("color").astype(np.float32)
    input_pn = pn[image_id, :, :, :]
    input_pn = torch.from_numpy(input_pn).float()
    input_pn = input_pn.to(device)
    input_v = v[image_id, :, :, :]
    input_v = torch.from_numpy(input_v).float()
    input_v = input_v.to(device)
    model.eval()
    with torch.no_grad():
        tim = time.time()
        pred_output = model(input_pn, input_v)
        diff = time.time() - tim
        print(diff * 1000)
    img = pred_output.cpu().detach().numpy()

    fig = plt.figure(figsize=(10, 7))
    #fig.add_subplot(1, 2, 1)
    plt.imshow(img[:, :, 0:3])
    #fig.add_subplot(1, 2, 2)
    #plt.imshow(color[image_id, :, :, 0:3])
    #plt.waitforbuttonpress()
    plt.savefig(f'{model_checkpoint_path}.png')

if __name__ == '__main__':
    load_and_vis('model/model_siren_500.pth')
