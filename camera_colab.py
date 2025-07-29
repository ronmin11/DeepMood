##INSTRUCTIONS: UNCOMMENT EVERYTHING (Ctrl A /)


# import base64
# from PIL import Image
# import numpy as np
# import torch
# import torchvision
# from torchvision import transforms
# from IPython.display import display, Javascript, update_display, Image as IPyImage
# from google.colab.output import register_callback, eval_js
# import io
# import time

# display_id = None

# class EmotionDetector:
#     def __init__(self, model_type=prediction_args['model_type'], checkpoint_path = get_best_model_path(), num_classes=7, verbose=False):
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

#         # Initialize MTCNN face detector
#         self.mtcnn = MTCNN(
#             keep_all=True,
#             min_face_size=40,
#             thresholds=[0.6, 0.7, 0.7],
#             device=self.device
#         )

#         self.model = self._load_model(model_type, checkpoint_path, num_classes)
#         self.transform = self._get_transform(model_type)
#         self.verbose = verbose

#         self.predictions_store = []
#         self.confidence_avg = []
    
#     def add(self, label, loc, val):
#         if label not in loc:
#             loc[label] = val
#         else:
#             loc[label] += val
    
#     def get_frequent_prediction(self):
#         highest = -100000
#         index = -1
#         for n, i in enumerate(self.predictions_store):
#           if n > highest:
#             highest = n
#             index = i
#         return {'prediction': self.predictions_store[index], 'confidence': self.confidence_avg[index] / self.predictions_store[index]} or None
            

#     def _load_model(self, model_type, checkpoint_path, num_classes):
#         if model_type == 'resnet50.a1_in1k' or model_type.startswith("google/vit"):
#             backbone = timm.create_model(model_type, pretrained=True, num_classes=0)
#             model = EmotionNet(backbone, num_classes)
#         else:
#             model = MoodCNN2(num_classes=num_classes)

#         model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
#         model.to(self.device)
#         model.eval()
#         return model

#     def _get_transform(self, model_type):
#         model_timm = timm.create_model(model_type, pretrained=True, num_classes=0)
#         config = timm.data.resolve_model_data_config(model_timm)
#         return transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#             # transforms.Normalize(config['mean'], config['std']),
#             transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
#         ])

#     def predict_emotion(self, image: Image.Image):
#         # Convert to RGB if needed
#         # if image.mode != 'RGB':
#         #     image = image.convert('RGB')

#         image = image.convert('L')
#         image = image.convert('RGB')  # Fake 3-channel so transform won't break

#         # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#         # Detect faces with MTCNN
#         boxes, _ = self.mtcnn.detect(image)

#         if boxes is None or len(boxes) == 0:
#             return "No face detected", None, None

#         # Process first detected face
#         x1, y1, x2, y2 = map(int, boxes[0])
#         w, h = x2 - x1, y2 - y1

#         # Expand bounding box by 20%
#         x1 = max(0, int(x1 - 0.2 * w))
#         y1 = max(0, int(y1 - 0.2 * h))
#         x2 = min(image.width, int(x2 + 0.2 * w))
#         y2 = min(image.height, int(y2 + 0.2 * h))

#         face_crop = image.crop((x1, y1, x2, y2))
#         face_tensor = self.transform(face_crop).unsqueeze(0).to(self.device)

#         with torch.no_grad():
#             output = self.model(face_tensor)
#             probs = torch.softmax(output, dim=1)
#             confidence, predicted = torch.max(probs, 1)
#             add(self.class_names[predicted.item()], self.predictions_store, 1)
#             add(self.class_names[predicted.item()], self.confidence_avg, confidence.item())
#             return self.class_names[predicted.item()], confidence.item(), (x1, y1, x2, y2)

#     def start_webcam(self):
#         # Clear previous webcam instances
#         display(Javascript('''
#             if (window.stream) {
#                 window.stream.getTracks().forEach(track => track.stop());
#             }
#             if (window.video) {
#                 window.video.remove();
#             }
#             if (window.canvas) {
#                 window.canvas.remove();
#             }
#         '''))

#         # Simplified webcam capture
#         js = """
#         async function capture() {
#             const video = document.createElement('video');
#             video.style.transform = 'scaleX(-1)';
#             document.body.appendChild(video);
#             video.width = 640;
#             video.height = 480;

#             const stream = await navigator.mediaDevices.getUserMedia({ video: true });
#             video.srcObject = stream;
#             await video.play();

#             const canvas = document.createElement('canvas');
#             canvas.width = video.width;
#             canvas.height = video.height;
#             const ctx = canvas.getContext('2d');
#             ctx.translate(canvas.width, 0);
#             ctx.scale(-1, 1);

#             function captureFrame() {
#                 ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
#                 const data = canvas.toDataURL('image/jpeg', 0.8);
#                 google.colab.kernel.invokeFunction('notebook.get_frame', [data], {});
#                 setTimeout(captureFrame, 100);
#             }
#             captureFrame();
#         }
#         capture();
#         """
#         display(Javascript(js))



# def get_frame(data):
#         global display_id
#         try:
#             # Decode base64 image
#             img_bytes = base64.b64decode(data.split(',')[1])
#             img_np = np.frombuffer(img_bytes, dtype=np.uint8)
#             frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

#             # Convert to PIL for processing
#             pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

#             # Predict emotion
#             emotion, conf, box = detector.predict_emotion(pil_image)

#             # Draw results on frame
#             if emotion != "No face detected" and box is not None:
#                 x1, y1, x2, y2 = box
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                 cv2.putText(frame, f"{emotion} ({conf:.2f})", (x1, y1-10),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
#             else:
#                 cv2.putText(frame, "No face detected", (10, 30),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

#             # Display image
#             _, buffer = cv2.imencode('.png', frame)
#             display_img = IPyImage(data=buffer.tobytes())

#             if display_id is None:
#                 display_id = display(display_img, display_id=True)
#             else:
#                 # update_display(display_img, display_id=display_id)
#                 display_id.update(display_img)

#         except Exception as e:
#             print(f"Error in get_frame: {str(e)}")

# if __name__ == "__main__":
#         # Initialize detector (adjust paths as needed)
#         detector = EmotionDetector(
#             # model_type=args.model_name,
#             prediction_args['model_type'],
#             num_classes=7,
#             verbose=False
#         )

#         # Test with a sample image first
#         try:
#             from google.colab import files
#             print("Upload a test image for verification")
#             uploaded = files.upload()
#             if uploaded:
#                 file_name = next(iter(uploaded))
#                 test_image = Image.open(file_name)

#                 # Display test image
#                 print(f"Testing with: {file_name}")
#                 display(test_image)

#                 # Predict emotion
#                 emotion, conf, box = detector.predict_emotion(test_image)
#                 print(f"Test Result: {emotion} (Confidence: {conf:.2f})")
#         except Exception as e:
#             print(f"Static test skipped: {e}")

#         # Start webcam
#         # register_callback('notebook.get_frame', get_frame)
#         # detector.start_webcam()