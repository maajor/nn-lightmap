import os
import torch

from torch import optim
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from loader import prepare_dataloader
from model import SirenGINet
from vis import load_and_vis
from torch.utils.tensorboard import SummaryWriter
from torch import nn

device = torch.device("cuda:0")
BATCH_SIZE = 1
TRAIN_EPOCHS = 300

train_loader, test_loader, img_shape = prepare_dataloader(batch_size=BATCH_SIZE, path='dataset/render_text_4k.npz')

loss_fn = torch.nn.MSELoss(reduction="sum")


def train_epoch(epoch, model, optimizer, writer):
    model.train()
    train_loss = 0
    for id, (p, n, v, uv, color) in enumerate(train_loader):
        n = n.to(device)
        v = v.to(device)
        p = p.to(device)
        optimizer.zero_grad()
        pred_output = model(p, n, v)
        color = color.to(device)
        loss = loss_fn(pred_output, color)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    train_loss = (
        train_loss
        * 255
        * 255
        / (len(train_loader.dataset) * img_shape[0] * img_shape[1] * img_shape[2])
    )
    writer.add_scalar("Loss/train", train_loss, epoch)
    print("====> Epoch: {} Average loss: {:.4f}".format(epoch, train_loss))
    return train_loss


def test_epoch(epoch, model):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for id, (p, n, v, uv, color) in enumerate(test_loader):
            n = n.to(device)
            v = v.to(device)
            p = p.to(device)
            pred_output = model(p, n, v)
            output = color.to(device)
            test_loss += loss_fn(pred_output, output).item()

    test_loss = (
        test_loss * 255 * 255 / (len(test_loader.dataset) * img_shape[0] * img_shape[1] * img_shape[2])
    )
    print("====> Epoch: {} Test set loss: {:.4f}".format(epoch, test_loss))
    return test_loss


def plot(train_loss, test_loss, save_name):
    iter_range = range(TRAIN_EPOCHS)
    plt.subplot(2, 1, 1)
    plt.plot(iter_range, np.log10(train_loss), "o-")
    plt.title("Train Loss vs. Epoches")
    plt.ylabel("Train Loss")
    plt.subplot(2, 1, 2)
    plt.plot(
        iter_range[int(len(test_loss) / 2) :],
        np.log10(np.array(test_loss))[int(len(test_loss) / 2) :],
        ".-",
    )
    plt.xlabel("Log Test Loss vs. Epoches")
    plt.ylabel("Log Test Loss Last Half")
    plt.savefig(save_name + ".png")
    plt.close()


def train(model, save_name, writer):
    train_loss = []
    test_loss = []
    model_path = "model/" + save_name + ".pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load("model/" + save_name + ".pth"))
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999))
    loop = tqdm.tqdm(range(TRAIN_EPOCHS))
    for epoch in loop:
        train_loss.append(train_epoch(epoch, model, optimizer, writer))
        test_loss.append(test_epoch(epoch, model))
        if epoch % 20 == 0:
            load_and_vis(model, f"{save_name}_{epoch}", epoch, writer)
        if epoch % 5 == 0:
            writer.flush()
        if epoch % 100 == 0:
            torch.save(model.state_dict(), f"model/{save_name}_{epoch}.pth")
    torch.save(model.state_dict(), "model/" + save_name + ".pth")
    plot(train_loss, test_loss, save_name)


def test(model, saved_name):
    model.load_state_dict(torch.load("model/" + saved_name + ".pth"))
    model = model.to(device)
    model.eval()
    test_epoch(0, model)


def train_all(lm_dim=256, lm_layer=5, dim_hidden=32, rf_dim=64, rf_layer=2):
    model = SirenGINet(lm_dim, lm_layer, dim_hidden, rf_dim, rf_layer)
    comment = f"{lm_dim}x{lm_layer}x{dim_hidden}x{rf_dim}x{rf_layer}"
    from datetime import datetime

    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    writer = SummaryWriter(
        comment=comment, log_dir=f"/root/tf-logs/runs_{current_time}_{comment}"
    )
    train(model, f"model_siren_{comment}", writer)
    writer.close()


if __name__ == "__main__":
    train_all(256, 5, 32, 64, 2)
