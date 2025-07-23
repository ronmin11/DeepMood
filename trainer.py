# trainer.py
import os, torch
import torch.nn as nn
import torch.nn.functional as F
from model import EmotionNet
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, clear_output
from sklearn.metrics import f1_score

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
            raise ValueError("Model not found")
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=args.lr_decay)
        self.criterion = nn.CrossEntropyLoss()

        # Logging DataFrame: logs everything you asked for
        self.logs_df = pd.DataFrame(columns=["Epoch", "Train Loss", "Train Accuracy", "Val Loss", "Val Accuracy", "F1 Score"])

    def train_network(self, epoch, loader, **kwargs):
        self.model.train()
        total_loss, correct, total = 0, 0, 0

        loop = tqdm(loader, desc=f"Epoch {epoch} [Training]", leave=False)
        for imgs, labels in loop:
            imgs, labels = imgs.to(self.device), labels.to(self.device)

            outputs = self.model(imgs)
            loss = self.criterion(outputs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            predicted = torch.argmax(outputs, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            loop.set_postfix(loss=loss.item())

        self.scheduler.step()
        avg_loss = total_loss / len(loader)
        accuracy = (correct / total) * 100.0
        return avg_loss, accuracy, self.optimizer.param_groups[0]['lr']

    def evaluate_network(self, epoch=0, loader=None, **kwargs):
        self.model.eval()
        #for avg_loss and accuracy
        correct, total, total_loss = 0, 0, 0

        #for f1 score
        all_preds, all_labels = [], []

        loop = tqdm(loader, desc=f"Epoch {epoch} [Evaluating]", leave=False)
        with torch.no_grad():
            for imgs, labels in loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                outputs = self.model(imgs)
                loss = self.criterion(outputs, labels)

                predicted = torch.argmax(outputs, dim=1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                total_loss += loss.item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                loop.set_postfix(accuracy=f"{(correct/total)*100:.2f}%")
                
        avg_loss = total_loss / len(loader)
        accuracy = (correct / total) * 100.0
        f1 = f1_score(all_labels, all_preds, average='weighted') * 100.0
        return avg_loss, accuracy, f1
    
    def log_metrics(self, epoch, train_loss, train_acc, val_loss, val_acc, f1):
        new_row = {
            "Epoch": epoch,
            "Train Loss": train_loss,
            "Train Accuracy": train_acc,
            "Val Loss": val_loss,
            "Val Accuracy": val_acc,
            "F1 Score": f1
        }
        self.logs_df = pd.concat([self.logs_df, pd.DataFrame([new_row])], ignore_index=True)
        display(self.logs_df.tail(10))  # display last 10 entries for readability in Colab

    def plot_metrics(self):
        sns.set(style="whitegrid")
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))

        # Plot Loss (Train + Validation)
        axes[0].set_title(f"Model Loss - {self.args.model_name}", fontsize=14)
        sns.lineplot(ax=axes[0], x="Epoch", y="Train Loss", data=self.logs_df, label="Train Loss", marker="o")
        sns.lineplot(ax=axes[0], x="Epoch", y="Val Loss", data=self.logs_df, label="Val Loss", marker="o")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()

        # Plot Accuracy + F1
        axes[1].set_title(f"Accuracy & F1 Score - {self.args.model_name}", fontsize=14)
        sns.lineplot(ax=axes[1], x="Epoch", y="Train Accuracy", data=self.logs_df, label="Train Accuracy", marker="o")
        sns.lineplot(ax=axes[1], x="Epoch", y="Val Accuracy", data=self.logs_df, label="Val Accuracy", marker="o")
        sns.lineplot(ax=axes[1], x="Epoch", y="F1 Score", data=self.logs_df, label="F1 Score", marker="o")

        # Highlight max points
        max_acc_idx = self.logs_df["Val Accuracy"].idxmax()
        max_f1_idx = self.logs_df["F1 Score"].idxmax()
        acc_x = self.logs_df.loc[max_acc_idx, "Epoch"]
        acc_y = self.logs_df.loc[max_acc_idx, "Val Accuracy"]
        f1_x = self.logs_df.loc[max_f1_idx, "Epoch"]
        f1_y = self.logs_df.loc[max_f1_idx, "F1 Score"]

        axes[1].plot(acc_x, acc_y, "ro")
        axes[1].text(acc_x, acc_y + 0.5, f"Max Val Acc: {acc_y:.2f}%", color="red")

        axes[1].plot(f1_x, f1_y, "go")
        axes[1].text(f1_x, f1_y + 0.5, f"Max F1: {f1_y:.2f}%", color="green")

        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Score (%)")
        axes[1].legend()

        plt.tight_layout()
        plt.show()

    def saveParameters(self, path):
        torch.save(self.model.state_dict(), path)

    def loadParameters(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
