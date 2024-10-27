import cvlib as cv
from cvlib.object_detection import draw_bbox
import cv2
import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
import albumentations
from PIL import Image

# Load label binarizer and model
print('Loading model and label binarizer...')
lb = joblib.load('lb.pkl')

class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.conv4 = nn.Conv2d(64, 128, 5)
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, len(lb.classes_))
        self.pool = nn.MaxPool2d(2, 2)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        bs, _, _, _ = x.shape
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

print('Model Loaded...')
model = CustomCNN()
model.load_state_dict(torch.load('model.pth', map_location='cpu'))
model.eval()
print('Loaded model state_dict...')

# Image augmentation (resize to match model input size)
aug = albumentations.Compose([
    albumentations.Resize(128, 128),  # Smaller size for faster CPU processing
])

# Colors for bounding boxes
colors = [
    (255, 0, 0),  # Blue
    (0, 255, 0),  # Green
    (0, 0, 255),  # Red (reserved for drowning)
    (255, 255, 0), # Cyan
    (255, 165, 0), # Orange
    (128, 0, 128), # Purple
]

def detectDrowning():
    # Video capture setup
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('Error: Unable to access the camera.')
        return

    # Initialize frame skip and confidence threshold
    frame_skip = 3  # Process every 3rd frame
    confidence_threshold = 0.6  # Ignore detections below this confidence level
    frame_count = 0
    
    with torch.no_grad():  # Disable gradients for faster performance
        while cap.isOpened():
            status, frame = cap.read()
            if not status:
                print('Error: Unable to capture video frame.')
                break

            frame_count += 1
            if frame_count % frame_skip != 0:
                continue  # Skip frame

            # Downscale frame early for faster detection
            small_frame = cv2.resize(frame, (640, 480))

            # Detect common objects in frame
            bbox, label, conf = cv.detect_common_objects(small_frame, confidence=confidence_threshold)

            # Process detected people
            for i, box in enumerate(bbox):
                if label[i] != 'person' or conf[i] < confidence_threshold:
                    continue

                # Resize image for model input
                pil_image = Image.fromarray(cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB))
                pil_image = aug(image=np.array(pil_image))['image']
                
                # Preprocess image for model
                pil_image = np.transpose(pil_image, (2, 0, 1)).astype(np.float32)
                pil_image = torch.tensor(pil_image, dtype=torch.float).unsqueeze(0).cpu()

                # Run model prediction
                outputs = model(pil_image)
                _, preds = torch.max(outputs.data, 1)
                pred_label = lb.classes_[preds]
                
                # Determine bounding box color
                isDrowning = (pred_label == 'drowning')
                box_color = (0, 0, 255) if isDrowning else colors[i % len(colors)]
                
                # Draw bounding box with label
                x1, y1, x2, y2 = box
                cv2.rectangle(small_frame, (x1, y1), (x2, y2), box_color, 2)
                label_text = f"ID:{i+1} {pred_label} ({conf[i]*100:.1f}%)"
                cv2.putText(small_frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
                print(f"Detection {i+1}: {pred_label}, Confidence: {conf[i]:.2f}")

            # Display the processed frame
            cv2.imshow("Real-time Drowning Detection", small_frame)

            # Quit if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

# Start detection
detectDrowning()
