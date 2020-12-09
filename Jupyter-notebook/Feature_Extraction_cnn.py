import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=(3, 3),
                        stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1)),

            nn.Conv2d(16, 32, kernel_size=(3, 3),
                        stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1)),

            nn.Conv2d(32, 8, kernel_size=(3, 3),
                        stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1)),

            nn.Dropout(p=0.2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(8 * 22 * 22, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 2)
        )

    def forward(self, x):
        x = self.features(x)
#         print(x.shape)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        probas = F.softmax(logits, dim=1)
        return logits, probas


PATH = './mask_net.pth'
net = Net()
net.load_state_dict(torch.load(PATH))


class LayerActivations:
    features = None

    def __init__(self, model, layer_num):
        self.hook = model[layer_num].register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output.cpu()

    def remove(self):
        self.hook.remove()

print(net.features)



# Model parameters
LEARNING_RATE = 1e-3
BATCH_SIZE = 8
EPOCH = 20
CLASS_NUM = 2

IMAGE_HEIGHT = 180
IMAGE_WIDTH = 180

data_dir = '../dataset'


# Load dataset
Transforms = transforms.Compose([
    transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = torchvision.datasets.ImageFolder(
    root=data_dir,
    transform=Transforms
)

for imgs in dataset:
    img = imgs[0].detach().numpy()
    img = torch.Tensor([img]*BATCH_SIZE)
    conv_out = LayerActivations(net.features, 8)
    o = net(img)
    conv_out.remove()
    act = conv_out.features
    act = torch.flatten(act, 1)
    img = act.detach().numpy()
    print(img)
    break
    