# model.py
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from urllib.request import urlopen
import timm

from tqdm import tqdm
import time

#Using pretrained resnet50.a1_in1k model as a feature extractor, without classifier head
backbone_model = timm.create_model(
    'resnet50.a1_in1k',
    pretrained=True,
    num_classes = 7,
)

# effNet = timm.create_model(
#     "efficientnetv2_m.in21k_ft_in1k",
#     pretrained=True,
#     num_classes = 7
# )


class MoodCNN1(nn.Module):
    def __init__(self, num_classes=7):
        super(MoodCNN1, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 112x112

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 56x56

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

# Increase capacity and add residual connections
class MoodCNN2(nn.Module):
    def __init__(self, num_classes=7):
        super(MoodCNN2, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.res1 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128)
        )

        self.conv2 = nn.Sequential(
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.res2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256)
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        res = x
        x = self.res1(x)
        x += F.interpolate(res, scale_factor=1, mode='nearest')  # Skip connection
        x = self.conv2(x)
        res = x
        x = self.res2(x)
        x += res  # Skip connection
        return self.classifier(x)


class EmotionNet(nn.Module):
    def __init__(self, backbone, num_classes):
        super(EmotionNet, self).__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(backbone.num_features, num_classes)
        # self.classifier = nn.Sequential(
        #     nn.Dropout(0.5),
        #     nn.Linear(backbone.num_features, 512),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.5),
        #     nn.Linear(512, num_classes)
        # )

    def forward(self, x):
        x = self.backbone(x)
        return self.classifier(x)