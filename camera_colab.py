import cv2
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN
from torchvision import transforms
import timm

from IPython.display import display, Javascript
# from google.colab.output import register_callback #UNCOMMENT THIS LINE IN GOOGLE COLAB
from model import EmotionNet, MoodCNN2 #COMMENT THIS LINE IN GOOGLE COLAB
from main import get_args #COMMENT THIS LINE IN GOOGLE COLAB
args = get_args() #COMMENT THIS LINE IN GOOGLE COLAB
import base64
from IPython.display import clear_output
import time

from IPython.display import display, update_display
from IPython.display import Image as IPyImage
import io

display_id = None  # Global variable to reuse display slot

class EmotionDetector:
    def __init__(self, model_type='resnet50.a1_in1k', checkpoint_path = f'{args.modelSavePath}/best.model', num_classes=7):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = model_type
        self.class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

        # Initialize face detector
        self.face_detector = MTCNN(keep_all=True, device=self.device)

        # Load model
        self.model = self._load_model(model_type, checkpoint_path, num_classes)
        self.transform = self._get_transform(model_type)

    def _load_model(self, model_type, checkpoint_path, num_classes):
        if model_type == 'resnet50.a1_in1k':
            backbone = timm.create_model(model_type, pretrained=False, num_classes=0)
            model = EmotionNet(backbone, num_classes)
        else:
            model = MoodCNN2(num_classes=num_classes)

        model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        model = model.to(self.device)
        model.eval()
        return model

    def _get_transform(self, model_type):
        if model_type == 'resnet50.a1_in1k':
            model_timm = timm.create_model(model_type, pretrained=True, num_classes=0)
            data_config = timm.data.resolve_model_data_config(model_timm)
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(data_config['mean'], data_config['std']),
            ])
        else:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.5]*3, [0.5]*3),
            ])

    def predict_emotion(self, image):
        if isinstance(image, str):
            image = Image.open(image)

        boxes, _ = self.face_detector.detect(image)
        if boxes is None:
            return "No face detected", None

        box = boxes[0]
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1

        x1 = max(0, int(x1 - 0.2 * w))
        y1 = max(0, int(y1 - 0.2 * h))
        x2 = min(image.width, int(x2 + 0.2 * w))
        y2 = min(image.height, int(y2 + 0.2 * h))

        face = image.crop((x1, y1, x2, y2))
        face_tensor = self.transform(face).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(face_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            return self.class_names[predicted.item()], confidence.item()

    def start_webcam(self):
        js_code = """
        async function capture() {
            const video = document.createElement('video');
            document.body.appendChild(video);
            video.width = 640;
            video.height = 480;

            const stream = await navigator.mediaDevices.getUserMedia({video:true});
            video.srcObject = stream;
            await video.play();

            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            canvas.width = 640;
            canvas.height = 480;

            while (true) {
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                const data = canvas.toDataURL('image/jpeg', 0.8);
                google.colab.kernel.invokeFunction('notebook.get_frame', [data], {});
                await new Promise(resolve => setTimeout(resolve, 100));  // 10 FPS approx
            }
        }
        capture();
        """
        display(Javascript(js_code))

# ============ You define this OUTSIDE the class ===============

def get_frame(data):
    global display_id

    # Decode image from base64
    img_bytes = base64.b64decode(data.split(',')[1])
    img_np = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

    # Convert to RGB PIL Image for model
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Run emotion prediction
    emotion, confidence = detector.predict_emotion(pil_img)

    # Add label if face detected
    if emotion != "No face detected":
        label = f"{emotion} ({confidence:.2f})"
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Encode image back to PNG bytes
    _, buf = cv2.imencode('.png', frame)
    img_bytes = buf.tobytes()

    # Prepare IPython Image to display
    image = IPyImage(data=img_bytes)

    # Display or update in the same output cell
    if display_id is None:
        display_id = display(image, display_id=True)
    else:
        update_display(image, display_id=display_id)



# ===============================================================





if __name__ == "__main__":
    detector = EmotionDetector(
        model_type=args.model_name
    )

    # # Test on single image
    # emotion, confidence = detector.predict_emotion("test_image.jpg")
    # print(f"Predicted emotion: {emotion} (Confidence: {confidence:.2f})")

    # Start real-time detection from webcam
    # detector.real_time_detection()

    # Run this after defining your class
    # register_callback('notebook.get_frame', get_frame) #UNCOMMENT THIS LINE IN GOOGLE COLAB
    detector.start_webcam()