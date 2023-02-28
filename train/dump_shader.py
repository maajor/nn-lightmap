from model import SirenGINet
import torch

if __name__ == '__main__':
    model = SirenGINet(256, 5, 8, 32, 2)
    model.load_state_dict(torch.load("model/model_siren_256x5x8x32x2_2200.pth"))
    model.dump_shader()