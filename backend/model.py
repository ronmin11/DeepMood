from transformers import AutoModel, pipeline, AutoImageProcessor
import os
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from torchvision import transforms
from torchvision.datasets import ImageFoswlder
from tqdm.notebook import tqdm
from torch.utils.tensorboard import SummaryWriter
from transformers import TrainingArguments, Trainer, AutoModelForImageClassification, DefaultDataCollator
from torch.utils.data import Dataset
import timm
import wandb
import evaluate
from sklearn.metrics import accuracy_score
from google.colab import drive



# Wrap ImageFolder in a dataset that returns dicts
class HuggingFaceVisionDataset(Dataset):
    def __init__(self, image_folder_dataset):
        self.image_folder_dataset = image_folder_dataset

    def __len__(self):
        return len(self.image_folder_dataset)

    def __getitem__(self, idx):
        image, label = self.image_folder_dataset[idx]
        return {
            "pixel_values": image,
            "labels": label # Change key to 'labels' (plural)
        }
    
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)

    acc = accuracy_score(labels, predictions)
    return {
        "accuracy": acc,

    }


#os.environ['WANDB_PROJECT'] = 'DeepMood'
#wandb.login()


device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_name = "google/vit-base-patch16-224"
pipe = pipeline("image-classification", model="google/vit-base-patch16-224")

#transform data
data_config = timm.data.resolve_model_data_config(pipe) #applies correct input size, normalization, etc.
transforms = timm.data.create_transform(**data_config, is_training=False) #takes in the required input parameters for the model and creates a transformation pipeline of Compose

train_ds = ImageFolder(root='/content/train',transform=transforms)
test_ds=ImageFolder(root='/content/test',transform=transforms)

train_ds = HuggingFaceVisionDataset(train_ds)
test_ds = HuggingFaceVisionDataset(test_ds)

"""
print(train_ds)
labels = train_ds.dataset.classes
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[i] = label
"""

model = AutoModelForImageClassification.from_pretrained(
    model_name,
    num_labels=7,
    #id2label=id2label,
    #label2id=label2id,
    ignore_mismatched_sizes=True,  # Allows for loading models with different input sizesC
)

#Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=5,
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",

)

trainer = Trainer(
   model = model,
   args =  training_args,
   train_dataset=train_ds,
   eval_dataset=test_ds,
   compute_metrics=compute_metrics
)

trainer.train()

drive.mount('/content/drive')

# Save model and processor after training
save_path = "/content/drive/MyDrive/DeepMood/vit-finetuned"  # or wherever you want

model.save_pretrained(save_path)
image_processor = AutoImageProcessor.from_pretrained(model_name)
image_processor.save_pretrained(save_path)


from transformers import AutoModelForImageClassification, AutoImageProcessor
from PIL import Image
import torch

# Load model and processor
model_path = "/content/drive/MyDrive/DeepMood/vit-finetuned"
model = AutoModelForImageClassification.from_pretrained(model_path)
processor = AutoImageProcessor.from_pretrained(model_path)

# Load image
image = Image.open("test.jpg")  # replace with your test image

# Preprocess
inputs = processor(images=image, return_tensors="pt")

# Predict
model.eval()
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = logits.argmax(-1).item()

print("Predicted class:", predicted_class)
