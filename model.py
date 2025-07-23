

# transform = transforms.Compose([
#     #Resize all images to become in the shape of 224 x 224
#     transforms.Resize((224, 224)),
#     #Converts all image matrices into tensors
#     transforms.ToTensor(),

#     # transforms.Normalize([0.5], [0.5]) # if the image was black and white
#     transforms.Normalize([0.485, 0.456, 0.406],
#                      [0.229, 0.224, 0.225])
# ])

# train_dataset = datasets.ImageFolder(root='EmotionDataset/train', transform=transform)
# test_dataset = datasets.ImageFolder(root='EmotionDataset/test', transform=transform)

# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# #We don't need to shuffle the test dataset when testing cause it wouldn't affect the model; the model already trained on shuffled data.
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# print(train_dataset.class_to_idx)
# #Outputs all the labels {'angry': 0, 'disgusted': 1, 'fearful': 2, 'happy': 3, 'neutral': 4, 'sad': 5, 'surprised': 6}
# #So now when we design our model, we can use a softmax activation function (0-1) and pass that into our fully connection network (nn) and get the predicted label


#=========================

# import pandas as pd
# from torch.utils.data import Dataset
# from PIL import Image
# import os
# import torch
# from torch.utils.data import Dataset, DataLoader
# import torchvision
# from torchvision import transforms, datasets
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from urllib.request import urlopen
# import timm

# from tqdm import tqdm
# import time

# #Using pretrained resnet50.a1_in1k model as a feature extractor, without classifier head
# backbone_model = timm.create_model(
#     'resnet50.a1_in1k', 
#     pretrained=True,
#     num_classes = 7,
# )
# backbone_model = backbone_model.eval()

# # get model specific transforms (normalization, resize)
# data_config = timm.data.resolve_model_data_config(backbone_model)
# transform = timm.data.create_transform(**data_config, is_training=False)


# train_dataset = datasets.ImageFolder(root='EmotionDataset/train', transform=transform)
# test_dataset = datasets.ImageFolder(root='EmotionDataset/test', transform=transform)

# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
# #We don't need to shuffle the test dataset when testing cause it wouldn't affect the model; the model already trained on shuffled data.
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# print(train_dataset.class_to_idx)

# class EmotionNet(nn.Module):
#     def __init__(self, backbone_model, n_classes):
#         super().__init__()
#         self.backbone_model = backbone_model
#         self.head = nn.Sequential(
#             nn.Linear(backbone_model.num_features, 512),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(512, n_classes)
#         )
#     def forward(self, x):
#         feats = self.backbone_model(x)
#         return self.head(feats)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = EmotionNet(backbone_model, n_classes=7).to(device)  #7 mood classes

# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(
#     model.parameters(), 
#     lr=1e-8, 
#     weight_decay=0,
# )


# for epoch in tqdm(range())





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


class MoodCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(MoodCNN, self).__init__()
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
    

class EmotionNet(nn.Module):
    def __init__(self, backbone, num_classes):
        super(EmotionNet, self).__init__()
        self.backbone = backbone
        # self.classifier = nn.Linear(backbone.num_features, num_classes)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(backbone.num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        return self.classifier(x)