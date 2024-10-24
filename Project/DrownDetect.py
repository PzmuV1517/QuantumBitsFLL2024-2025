import cv2
import torch
import time
import joblib
import numpy as np
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from yolov5 import YOLOv5

# Load label binarizer and model
print('Loading model and label binarizer...')
lb = joblib.load('lb.pkl')

# Define your custom CNN model for drowning detection
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

# Load the custom drowning detection model
print('Model Loaded...')
model = CustomCNN()
model.load_state_dict(torch.load('model.pth', map_location='cpu'))  # Load to CPU (modify if using GPU)
model.eval()
print('Loaded model state_dict...')

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Preprocessing for the custom model
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load YOLOv5 model (from local weights to avoid external dependencies)
yolo = YOLOv5("/home/kali/QuantumBitsFLL2024-2025/Project/yolov5n.pt", device=device)  # Specify the correct path to your model file

def detectDrowning():
    isDrowning = False
    frame_count = 0
    skip_frames = 2  # Skip every 2 frames for better FPS
    frame_width, frame_height = 640, 480  # Resize frames for faster processing

    # Access the camera feed
    cap = cv2.VideoCapture(0)  # Use camera index 0 (default camera)
    
    if not cap.isOpened():
        print('Error: Unable to access the camera.')
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    while cap.isOpened():
        status, frame = cap.read()

        if not status:
            print('Error: Unable to capture video frame.')
            break

        # Skip frames to improve FPS
        if frame_count % skip_frames == 0:
            # Resize frame to reduce computation
            frame_resized = cv2.resize(frame, (frame_width, frame_height))

            # Perform YOLOv5 object detection
            results = yolo.predict(frame_resized)

            bbox = []
            labels = []
            confs = []

            for pred in results.pred[0]:  # Iterate through the predicted results
                if yolo.names[int(pred[-1])] == 'person':  # Detect only people
                    bbox.append([int(pred[0]), int(pred[1]), int(pred[2]), int(pred[3])])
                    labels.append('person')
                    confs.append(float(pred[4]))

            # If only one person is detected, use model-based detection
            if len(bbox) == 1:
                bbox0 = bbox[0]

                with torch.no_grad():
                    # Preprocess image
                    pil_image = Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))
                    pil_image = preprocess(pil_image).unsqueeze(0).to(device)  # Send to GPU if available
                    
                    outputs = model(pil_image)
                    _, preds = torch.max(outputs.data, 1)

                print("Swimming status:", lb.classes_[preds])
                isDrowning = lb.classes_[preds] == 'drowning'

                # Draw bounding box and label on the frame
                for box in bbox:
                    cv2.rectangle(frame_resized, (box[0], box[1]), (box[2], box[3]), (0, 255, 0) if not isDrowning else (0, 0, 255), 2)
                    label = "Drowning" if isDrowning else "Safe"
                    cv2.putText(frame_resized, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0) if not isDrowning else (0, 0, 255), 2)

            # If more than one person is detected, use logic-based detection
            elif len(bbox) > 1:
                centres = [[(box[0] + box[2]) / 2, (box[1] + box[3]) / 2] for box in bbox]

                distances = [
                    np.sqrt((centres[i][0] - centres[j][0]) ** 2 + (centres[i][1] - centres[j][1]) ** 2)
                    for i in range(len(centres))
                    for j in range(i + 1, len(centres))
                ]

                isDrowning = len(distances) > 0 and min(distances) < 50

                for box in bbox:
                    cv2.rectangle(frame_resized, (box[0], box[1]), (box[2], box[3]), (0, 255, 0) if not isDrowning else (0, 0, 255), 2)
                    label = "Drowning" if isDrowning else "Safe"
                    cv2.putText(frame_resized, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0) if not isDrowning else (0, 0, 255), 2)

            # Display the output frame
            cv2.imshow("Real-time Drowning Detection", frame_resized)

        frame_count += 1

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()

# Start drowning detection using the camera feed
detectDrowning()
