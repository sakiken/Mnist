# 搭建神经网络
import torch
from torch import nn


class Tudui(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x


# main函数
if __name__ == '__main__':
    tudui = Tudui()
    # 验证网络模型的正确性，创造一个输入尺寸，判断输出尺寸是不是我们想要的
    input = torch.ones((64, 3, 32, 32))  # batchsize=64,channel=3,尺寸32*32
    output = tudui(input)
    print(output.shape)