import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets,models,transforms

model = models.vgg16(pretrained=True)

print(f"vgg16 = {model}")
