# trainer.py
import os, torch
import torch.nn as nn
import torch.nn.functional as F
from model import EmotionNet
from tqdm import tqdm

import timm
from model import MoodCNN  # <- this is your custom model, no timm


class Trainer:
    def __init__(self, args, epoch=1):
        self.args = args
        self.epoch = args.epochs

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if args.model_name == 'custom':
            self.model = MoodCNN(num_classes=args.num_classes).to(self.device)
        elif args.model_name == 'resnet50.a1_in1k':
            backbone = timm.create_model(args.model_name, pretrained=True, num_classes=0)
            self.model = EmotionNet(backbone, args.num_classes).to(self.device)
        else:
            return ValueError("Model not found")
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=args.lr_decay)
        self.criterion = nn.CrossEntropyLoss()

    def train_network(self, epoch, loader, **kwargs):
        self.model.train()
        total_loss = 0

        loop = tqdm(loader, desc=f"Epoch {epoch} [Training]", leave=False)
        for imgs, labels in loop:
            imgs, labels = imgs.to(self.device), labels.to(self.device)

            outputs = self.model(imgs)
            loss = self.criterion(outputs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        self.scheduler.step()
        avg_loss = total_loss / len(loader)
        return avg_loss, self.optimizer.param_groups[0]['lr']

    def evaluate_network(self, epoch=0, loader=None, **kwargs):
        self.model.eval()
        correct = 0
        total = 0

        loop = tqdm(loader, desc=f"Epoch {epoch} [Evaluating]", leave=False)
        with torch.no_grad():
            for imgs, labels in loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                outputs = self.model(imgs)
                predicted = torch.argmax(outputs, dim=1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                loop.set_postfix(accuracy=f"{(correct/total)*100:.2f}%")
                
        mAP = (correct / total) * 100
        return mAP

    def saveParameters(self, path):
        torch.save(self.model.state_dict(), path)

    def loadParameters(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
