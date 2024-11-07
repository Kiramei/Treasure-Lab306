import torch
from torch import nn


class Model(nn.Module):
    """
    LeNet-5 model
    Model architecture: C1 -> S2 -> C3 -> S4 -> C5 -> F6 -> Output
    """

    def __init__(self):
        super(Model, self).__init__()
        self.init_pool = nn.AvgPool2d(kernel_size=4)
        self.c1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1)
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(True)

        self.c2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1)
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(32 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, src):
        src = self.init_pool(src)
        src = self.c1(src)
        src = self.relu(src)
        src = self.max_pool1(src)
        src = self.c2(src)
        src = self.relu(src)
        src = self.max_pool2(src)
        src = torch.flatten(src, 1)
        src = self.fc1(src)
        src = self.relu(src)
        src = self.fc2(src)
        src = self.relu(src)
        src = self.fc3(src)
        return src


if __name__ == "__main__":
    x = torch.rand([1, 3, 128, 128])
    model = Model()
    y = model(x)
    print(y.shape)
