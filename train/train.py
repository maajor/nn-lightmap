import os
import torch

from torch import optim
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from loader import prepare_dataloader
from model import SirenGINet
from vis import load_and_vis

device = torch.device("cuda:0")
BATCH_SIZE = 1
TRAIN_EPOCHS = 3000

train_loader, test_loader, img_shape = prepare_dataloader(batch_size=BATCH_SIZE)

loss_fn = torch.nn.MSELoss(reduction="sum")


def train_epoch(epoch, model, optimizer):
    model.train()
    train_loss = 0
    for id, (pn, v, color) in enumerate(train_loader):
        pn = pn.to(device)
        v = v.to(device)
        optimizer.zero_grad()
        pred_output  = model(pn, v)
        color = color.to(device)
        loss = loss_fn(pred_output, color)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    train_loss = (
        train_loss * 255 / (len(train_loader.dataset) * (img_shape[0] * img_shape[1]))
    )
    print("====> Epoch: {} Average loss: {:.4f}".format(epoch, train_loss))
    return train_loss


def test_epoch(epoch, model):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for id, (pn, v, color) in enumerate(test_loader):
            pn = pn.to(device)
            v = v.to(device)
            pred_output = model(pn, v)
            output = color.to(device)
            test_loss += loss_fn(pred_output, output).item()

    test_loss = (
        test_loss * 255 / (len(test_loader.dataset) * (img_shape[0] * img_shape[1]))
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


def train(model, save_name):
    train_loss = []
    test_loss = []
    model_path = "model/" + save_name + ".pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load("model/" + save_name + ".pth"))
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999))
    loop = tqdm.tqdm(range(TRAIN_EPOCHS))
    for epoch in loop:
        train_loss.append(train_epoch(epoch, model, optimizer))
        test_loss.append(test_epoch(epoch, model))
        if epoch % 100 == 0:
            torch.save(model.state_dict(), f"model/{save_name}_{epoch}.pth")
            load_and_vis(model, f"{save_name}_{epoch}")
    torch.save(model.state_dict(), "model/" + save_name + ".pth")
    plot(train_loss, test_loss, save_name)


def test(model, saved_name):
    model.load_state_dict(torch.load("model/" + saved_name + ".pth"))
    model = model.to(device)
    model.eval()
    test_epoch(0, model)


def train_all():
    dim_hidden = 64
    num_layers = 5
    model = SirenGINet(dim_hidden, num_layers)
    train(model, f"model_siren_{dim_hidden}_{num_layers}")


if __name__ == "__main__":
    train_all()
