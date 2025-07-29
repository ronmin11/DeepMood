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
# backbone_model = timm.create_model(
#     'resnet50.a1_in1k',
#     pretrained=True,
#     num_classes = 0,
# )

# effNet = timm.create_model(
#     "efficientnetv2_m.in21k_ft_in1k",
#     pretrained=True,
#     num_classes = 0
# )

#Vision Transformer for > 5k images
# backbone_model = timm.create_model(
#     'google/vit_base_patch16_224',
#     pretrained=True,
#     num_classes = 0,
# )

#ViT pretrained on 14 million images
# backbone_model = timm.create_model(
#     'vit_base_patch16_224_in21k',
#     pretrained=True,
#     num_classes = 0,
# )

# backbone_model = timm.create_model(
#     'vit_base_patch16_224.augreg_in21k',
#     pretrained=True,
#     num_classes = 0,
# )

#============================================

models = [
    'resnet50.a1_in1k', # Used multiple times, got max 67% accuracy
    'efficientnetv2_m.in21k_ft_in1k', #Never tested
    'vit_base_patch16_224_in21k', # Deprecated
    'google/vit_base_patch16_224', #used once, got 72.14% accuracy
    'vit_base_patch16_224.augreg_in21k' #Testing now
]
model_name = models[4]
num_classes = 7

def get_model_info():
    return model_name, num_classes

backbone_model = timm.create_model(
    model_name,
    pretrained=True,
    num_classes = 0,
)



#STEPS
# 1) List initial distributions of each emotion from the original data  DONE
# 2) Use sklearn for f1 score and weighted accuracy  DONE
# 3) Find the predictions for each label and see which one is the worst and best. Consider removing the one label that performs the worst. ???



class MoodCNN1(nn.Module):
    def __init__(self, num_classes=num_classes):
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
    def __init__(self, num_classes=num_classes):
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
        self.classifier = nn.Sequential(
            # nn.Dropout(0.3),
            # nn.Linear(backbone.num_features, 512),
            # nn.ReLU(inplace=True),
            # nn.Dropout(0.3),
            # nn.Linear(512, num_classes)

            nn.BatchNorm1d(backbone.num_features),
            nn.Dropout(0.5),
            nn.Linear(backbone.num_features, 512),
            nn.SiLU(inplace=True),  # Better activation
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        return self.classifier(x)