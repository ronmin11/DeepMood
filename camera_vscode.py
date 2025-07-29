import base64
import io
import time
import os
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from facenet_pytorch import MTCNN
import timm
from model import EmotionNet, MoodCNN2
from test_single_image_url import get_prediction_args

display_id = None

class EmotionDetector:
    def __init__(self, model_type=get_prediction_args()['model_type'], checkpoint_path = get_prediction_args()['checkpoint_path'], num_classes=7, verbose=False):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        self.verbose = verbose

        self.mtcnn = MTCNN(keep_all=True, min_face_size=40, device=self.device)
        self.model = self._load_model(model_type, checkpoint_path, num_classes)
        self.transform = self._get_transform(model_type)

        self.predictions_store = []
        self.confidence_avg = []

    def add(self, label, loc, val):
        if label not in loc:
            loc[label] = val
        else:
            loc[label] += val
    
    def get_frequent_prediction(self):
        highest = -100000
        index = -1
        for n, i in enumerate(self.predictions_store):
          if n > highest:
            highest = n
            index = i
        return {'prediction': self.predictions_store[index], 'confidence': self.confidence_avg[index] / self.predictions_store[index]} or None

    def _load_model(self, model_type, checkpoint_path, num_classes):
        if model_type.startswith('resnet') or model_type.startswith("google/vit"):
            backbone = timm.create_model(model_type, pretrained=True, num_classes=0)
            model = EmotionNet(backbone, num_classes)
        else:
            model = MoodCNN2(num_classes=num_classes)

        model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model

    def _get_transform(self, model_type):
        model_timm = timm.create_model(model_type, pretrained=True, num_classes=0)
        config = timm.data.resolve_model_data_config(model_timm)
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # transforms.Normalize(config['mean'], config['std']),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def predict_emotion(self, pil_img: Image.Image):
        # image = pil_img.convert('RGB')
        image = pil_img.convert('L')
        image = image.convert('RGB')  # Fake 3-channel so transform won't break
        
        # Detect faces with MTCNN
        boxes, _ = self.mtcnn.detect(image)

        if boxes is None or len(boxes) == 0:
            return "No face detected", None, None

        # Process first detected face
        x1, y1, x2, y2 = map(int, boxes[0])
        w, h = x2 - x1, y2 - y1

        # Expand bounding box by 20%
        x1 = max(0, int(x1 - 0.2 * w))
        y1 = max(0, int(y1 - 0.2 * h))
        x2 = min(image.width, int(x2 + 0.2 * w))
        y2 = min(image.height, int(y2 + 0.2 * h))

        face_crop = image.crop((x1, y1, x2, y2))
        face_tensor = self.transform(face_crop).unsqueeze(0).to(self.device)

        # Make and store predictions
        with torch.no_grad():
            output = self.model(face_tensor)
            probs = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probs, 1)
            self.add(self.class_names[predicted.item()], self.predictions_store, 1)
            self.add(self.class_names[predicted.item()], self.confidence_avg, confidence.item())
            return self.class_names[predicted.item()], confidence.item(), (x1, y1, x2, y2)

    def run_webcam(self):
        cap = cv2.VideoCapture(0)
        print("Starting webcam...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert to PIL
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(img_rgb)

            emotion, conf, box = self.predict_emotion(pil_image)

            if box is not None:
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{emotion} ({conf:.2f})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "No face detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            cv2.imshow('Emotion Detector', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()