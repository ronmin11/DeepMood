import torch
import requests
from PIL import Image
from torchvision import transforms
from model import EmotionNet, MoodCNN2  # UNCOMMENT IN VSCODE
import timm
import io
import os
import sys

prediction_args = {
    # "model_type": 'resnet50.a1_in1k',
    "model_type": 'google/vit_base_patch16_224',
    "checkpoint_path": 'best.model',
    "class_names": ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'],
}

def get_prediction_args():
    return prediction_args

# Step 3: Load model (same constructor and checkpoint path)
# Moved this part of code to the outside to speed it up
model = timm.create_model(prediction_args['model_type'], pretrained=True, num_classes=0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backbone = timm.create_model(prediction_args['model_type'], pretrained=False, num_classes=0)
model = EmotionNet(backbone, num_classes=len(prediction_args['class_names']))
model.load_state_dict(torch.load(prediction_args['checkpoint_path'], map_location=device))
model.to(device)
model.eval()

def load_and_transform_image(path):
    image = Image.open(path).convert("L").convert("RGB")  # Ensure 3 channels
    data_config = timm.data.resolve_model_data_config(backbone)

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(data_config['mean'], data_config['std']),
    ])
    return transform(image).unsqueeze(0)

def main(image_path):
    if not os.path.exists(image_path):
        print(f"Error: Image not found -> {image_path}")
        return

    image_tensor = load_and_transform_image(image_path).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probs, dim=1)
        predicted_label = prediction_args['class_names'][predicted.item()]

    print(f"Predicted Emotion: {predicted_label} ({confidence.item():.2f})")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python detect_emotion.py path/to/image.jpg")
    else:
        main(sys.argv[1])