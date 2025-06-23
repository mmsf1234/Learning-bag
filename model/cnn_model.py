import torch
import torch.nn as nn

class CatDogCNN(nn.Module):
    def __init__(self):
        super(CatDogCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),  # 输入: 3x128x128
            nn.ReLU(),
            nn.MaxPool2d(2),                 # 输出: 16x64x64

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)                  # 输出: 32x32x32
        )

        # 关键：动态获取 flatten 后维度
        self.flatten_dim = self._get_flatten_size()

        self.fc = nn.Sequential(
            nn.Linear(self.flatten_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def _get_flatten_size(self):
        # 随便构造了一个输入，推理经过 conv 后的维度
        with torch.no_grad():
            x = torch.zeros(1, 3, 128, 128)  # 假设输入图像大小为128x128
            x = self.conv(x)
            return x.view(1, -1).shape[1]

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
