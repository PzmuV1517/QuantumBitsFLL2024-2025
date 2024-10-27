import cvlib as cv
from cvlib.object_detection import draw_bbox
import cv2
import time
import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
import albumentations
from torch.utils.data import Dataset, DataLoader
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

aug = albumentations.Compose([
    albumentations.Resize(224, 224),
])

# Define colors for bounding boxes for different individuals
colors = [
    (255, 0, 0),  # Blue
    (0, 255, 0),  # Green
    (0, 0, 255),  # Red (reserved for drowning)
    (255, 255, 0), # Cyan
    (255, 165, 0), # Orange
    (128, 0, 128), # Purple
]

def detectDrowning():
    isDrowning = False
    fram = 0
    
    # Use camera feed instead of video file
    cap = cv2.VideoCapture(0)  # Use camera index 0 (default camera)
    
    if not cap.isOpened():
        print('Error: Unable to access the camera.')
        return

    while cap.isOpened():
        status, frame = cap.read()

        if not status:
            print('Error: Unable to capture video frame.')
            break

        # Apply object detection
        bbox, label, conf = cv.detect_common_objects(frame)

        # If people are detected
        if len(bbox) > 0:
            for i, box in enumerate(bbox):
                # Get color for the bounding box, reserving red for drowning
                box_color = colors[i % len(colors)] if lb.classes_[0] != 'drowning' else (0, 0, 255)

                # Convert the frame for model input
                with torch.no_grad():
                    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    pil_image = aug(image=np.array(pil_image))['image']
                    
                    pil_image = np.transpose(pil_image, (2, 0, 1)).astype(np.float32)
                    pil_image = torch.tensor(pil_image, dtype=torch.float).cpu()
                    pil_image = pil_image.unsqueeze(0)
                    
                    outputs = model(pil_image)
                    _, preds = torch.max(outputs.data, 1)
                
                # Get the prediction label
                pred_label = lb.classes_[preds]
                print(f"Detection {i+1} - Label: {pred_label}, Confidence: {conf[i]:.2f}, Box: {box}")
                
                # Set drowning flag
                isDrowning = (pred_label == 'drowning')
                box_color = (0, 0, 255) if isDrowning else colors[i % len(colors)]

                # Draw bounding box and label
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                label_text = f"ID:{i+1} {pred_label} ({conf[i]*100:.1f}%)"
                cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

        else:
            print("No persons detected in this frame.")

        # Display the output frame
        cv2.imshow("Real-time Drowning Detection", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()

# Start drowning detection using the camera feed
detectDrowning()
