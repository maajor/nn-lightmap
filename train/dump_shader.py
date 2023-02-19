from model import SirenGINet
import torch

if __name__ == '__main__':
    model = SirenGINet(256, 5, 32, 32, 3)
    model.load_state_dict(torch.load("model/model_siren_256x5x32x32x3_2400.pth"))
    model.dump_shader()