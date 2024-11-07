import torch
from torch import nn


class Model(nn.Module):
    """
    LeNet-5 model
    Model architecture: C1 -> S2 -> C3 -> S4 -> C5 -> F6 -> Output
    """

    def __init__(self):
        super(Model, self).__init__()
        self.c1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, padding=2)
        self.act = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.c2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.c3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4)
        self.c4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4)
        self.c5 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5)
        self.flatten = nn.Flatten()
        self.f1 = nn.Linear(32, 16)
        self.output = nn.Linear(16, 2)

    def forward(self, src):
        src = self.pool(self.act(self.c1(src)))
        src = self.pool(self.act(self.c2(src)))
        src = self.pool(self.act(self.c3(src)))
        src = self.pool(self.act(self.c4(src)))
        src = self.act(self.c5(src))
        src = self.flatten(src).unsqueeze(1)
        src = self.act(self.f1(src))
        src = self.output(src).squeeze(1)
        return src.squeeze(1)


if __name__ == "__main__":
    x = torch.rand([1, 3, 128, 128])
    model = Model()
    y = model(x)
    print(y.shape)
