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

def detectDrowning():
    isDrowning = False
    frame = 0
    
    # Use camera feed instead of video file
    cap = cv2.VideoCapture(0)  # Use camera index 0 (default camera)
    cap.set(cv2.CAP_PROP_FPS, 30)

    
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

        # If only one person is detected, use model-based detection
        if len(bbox) == 1:
            bbox0 = bbox[0]
            centre = [(bbox0[0] + bbox0[2]) / 2, (bbox0[1] + bbox0[3]) / 2]

            with torch.no_grad():
                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                pil_image = aug(image=np.array(pil_image))['image']
                
                pil_image = np.transpose(pil_image, (2, 0, 1)).astype(np.float32)
                pil_image = torch.tensor(pil_image, dtype=torch.float).cpu()
                pil_image = pil_image.unsqueeze(0)
                
                outputs = model(pil_image)
                _, preds = torch.max(outputs.data, 1)

            print("Swimming status:", lb.classes_[preds])
            if lb.classes_[preds] == 'drowning':
                isDrowning = True
            else:
                isDrowning = False

            # Draw bounding box and label on the frame
            out = draw_bbox(frame, bbox, label, conf, isDrowning)

        # If more than one person is detected, use logic-based detection
        elif len(bbox) > 1:
            centres = [[(bbox[i][0] + bbox[i][2]) / 2, (bbox[i][1] + bbox[i][3]) / 2] for i in range(len(bbox))]

            distances = [
                np.sqrt((centres[i][0] - centres[j][0]) ** 2 + (centres[i][1] - centres[j][1]) ** 2)
                for i in range(len(centres))
                for j in range(i + 1, len(centres))
            ]

            if len(distances) > 0 and min(distances) < 50:
                isDrowning = True
            else:
                isDrowning = False

            out = draw_bbox(frame, bbox, label, conf, isDrowning)

        else:
            out = frame

        # Display the output frame
        cv2.imshow("Real-time Drowning Detection", out)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()

# Start drowning detection using the camera feed
detectDrowning()
