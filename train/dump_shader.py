from model import SirenGINet
import torch

if __name__ == '__main__':
    model = SirenGINet(256, 5, 16, 64, 2)
    model.load_state_dict(torch.load("model/model_siren_256x5x16x64x2.pth"))
    model.dump_shader()