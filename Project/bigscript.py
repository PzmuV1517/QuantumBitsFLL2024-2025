import time
import cvlib as cv
from cvlib.object_detection import draw_bbox
import cv2
import torch
import numpy as np
import joblib
from tellopy import Tello
from threading import Thread
from PIL import Image
import albumentations
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Custom CNN definition (same as in the original AI drowning detection code)
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

# Load AI model and label binarizer
lb = joblib.load('lb.pkl')
model = CustomCNN()
model.load_state_dict(torch.load('model.pth', map_location='cpu'))
model.eval()

aug = albumentations.Compose([
    albumentations.Resize(224, 224),
])

# Initialize Tello drone
drone = Tello()

# Path Planning Function: Capture the waypoints
def plan_path():
    print("Planning Path...")
    waypoints = []  # List to hold the waypoints (each is a tuple (x, y))
    # Capture waypoints (hardcoded for this example, you can integrate mouse input or GPS)
    # Example waypoints (x, y) positions (adjust to your desired coordinates):
    waypoints = [(0, 0), (50, 0), (50, 50), (0, 50), (0, 0)]
    return waypoints

# AI Drowning Detection Function: Runs in parallel during flight
def detectDrowning(frame):
    bbox, label, conf = cv.detect_common_objects(frame)
    isDrowning = False

    if len(bbox) == 1:
        # Convert frame to PIL image
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        pil_image = aug(image=np.array(pil_image))['image']
        pil_image = np.transpose(pil_image, (2, 0, 1)).astype(np.float32)
        pil_image = torch.tensor(pil_image, dtype=torch.float).cpu()
        pil_image = pil_image.unsqueeze(0)

        # Get prediction from model
        with torch.no_grad():
            outputs = model(pil_image)
            _, preds = torch.max(outputs.data, 1)
        
        print("Swimming status:", lb.classes_[preds])
        if lb.classes_[preds] == 'drowning':
            isDrowning = True

    return isDrowning, bbox, label, conf

# Drone Navigation: Fly the drone along the path and run AI simultaneously
def fly_path(waypoints):
    # Takeoff
    drone.takeoff()
    time.sleep(3)  # Allow time to stabilize after takeoff
    
    # Fly the planned waypoints
    for wp in waypoints:
        x, y = wp
        print(f"Flying to: {wp}")
        # Example movement commands based on the x, y coordinates
        drone.forward(20)  # Move forward by 20 cm (adjust based on your waypoints)
        time.sleep(2)  # Sleep to allow movement
        drone.right(20)  # Move right by 20 cm (adjust based on your path)
        time.sleep(2)

        # Capture frame from video stream and run AI detection
        cap = cv2.VideoCapture(0)  # Start video feed
        status, frame = cap.read()

        if not status:
            print('Error: Unable to capture video frame.')
            break
        
        isDrowning, bbox, label, conf = detectDrowning(frame)
        if isDrowning:
            print("Drowning detected! Taking action!")

        # Display the frame with bounding boxes
        out = draw_bbox(frame, bbox, label, conf, isDrowning)
        cv2.imshow("Drone Camera", out)

        # Quit display if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Return to the starting point
    print("Returning to start...")
    drone.backward(20)  # Example command to return to start (adjust accordingly)
    time.sleep(2)
    
    # Land
    print("Landing...")
    drone.land()

    # Close video capture
    cap.release()
    cv2.destroyAllWindows()

# Main Control Loop
def main():
    # Connect to Tello drone
    drone.connect()
    print("Connected to the drone...")

    while True:
        # Step 1: Plan the path
        waypoints = plan_path()

        # Step 2: Fly the path and run AI
        fly_path(waypoints)

        # Step 3: Wait 5 minutes (300 seconds) before repeating
        print("Waiting for 5 minutes...")
        time.sleep(300)

if __name__ == '__main__':
    main()
