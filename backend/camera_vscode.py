import cv2
import torch
import timm
import numpy as np
from PIL import Image
from torchvision import transforms
from facenet_pytorch import MTCNN

# Your model classes
from model import EmotionNet  # or your actual model file
# from model import MoodCNN2

class EmotionDetector:
    def __init__(self, model_path='checkpoints/best.model', model_type='resnet50.a1_in1k', num_classes=7):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = model_type
        self.class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

        # Load YOLOv11 (or MTCNN fallback)
        self.face_detector = MTCNN(keep_all=False, device=self.device)

        # Load emotion model
        self.model = self._load_model(model_path, model_type, num_classes)
        self.transform = self._get_transform(model_type)

    def _load_model(self, checkpoint_path, model_type, num_classes):
        if model_type == 'resnet50.a1_in1k':
            backbone = timm.create_model(model_type, pretrained=False, num_classes=0)
            model = EmotionNet(backbone, num_classes)
        else:
            raise ValueError("Unsupported model type")

        model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        model.to(self.device).eval()
        return model

    def _get_transform(self, model_type):
        model_timm = timm.create_model(model_type, pretrained=True, num_classes=0)
        data_config = timm.data.resolve_model_data_config(model_timm)
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(data_config['mean'], data_config['std']),
        ])

    def predict_emotion(self, face_image):
        tensor = self.transform(face_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model(tensor)
            prob = torch.softmax(out, dim=1)
            confidence, pred = torch.max(prob, 1)
            return self.class_names[pred.item()], confidence.item()


def main():
    detector = EmotionDetector()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Webcam not found")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB for MTCNN
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)

        boxes, _ = detector.face_detector.detect(pil_image)

        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                face = pil_image.crop((x1, y1, x2, y2))

                emotion, conf = detector.predict_emotion(face)
                label = f"{emotion} ({conf:.2f})"

                # Draw bounding box & label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 255, 0), 2)

        cv2.imshow("Emotion Detection", frame)

        # Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
