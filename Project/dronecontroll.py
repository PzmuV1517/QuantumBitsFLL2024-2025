import json
import math
from djitellopy import Tello
import pygame
import cv2
import time
import threading
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import albumentations
import cvlib as cv
from cvlib.object_detection import draw_bbox

# Initialize Pygame
pygame.init()

# Constants for Pygame window
WINDOW_WIDTH = 960
WINDOW_HEIGHT = 720
WINDOW_TITLE = "Drone Camera Feed"

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

# Check if GPU is available and set device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

print('Model Loaded...')
model = CustomCNN().to(device)
model.load_state_dict(torch.load('model.pth', map_location=device))
model.eval()
print('Loaded model state_dict...')

aug = albumentations.Compose([
    albumentations.Resize(224, 224),
])

# Function to move drone based on calculated distances and angles
def move_to_waypoint(drone, wp):
    for point in wp:
        distance_cm = point['dist_cm']
        angle_deg = point['angle_deg']

        # Rotate to the angle
        print(f"Rotating {angle_deg} degrees")
        drone.rotate_clockwise(angle_deg)
        time.sleep(2)

        # Move forward the specified distance
        print(f"Moving forward {distance_cm} cm")
        drone.move_forward(distance_cm)
        time.sleep(2)

# Function to convert OpenCV frame to Pygame surface
def frame_to_surface(frame):
    # Convert BGR (OpenCV format) to RGB (Pygame format) COMMENT TO REMOVE IF UNNECESARY!!!!!!!!!
    #frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Transpose the frame to match Pygame's (width, height) format
    surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
    return surface

# Main program
def main():
    # Load the JSON file
    with open("waypoint.json", "r") as file:
        data = json.load(file)

    wp = data['wp']  # Waypoints

    # Connect to Tello drone
    drone = Tello()
    drone.connect()

    # Start the video stream
    drone.streamon()

    # Create a Pygame window
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE)
    pygame.display.set_caption(WINDOW_TITLE)

    # Check battery
    battery = drone.get_battery()
    print(f"Battery: {battery}%")
    if battery < 40:
        print("Battery too low for flight. Please charge the drone.")
        return

    # Take off
    print("Taking off...")
    drone.takeoff()
    time.sleep(5)

    # Fly to 1 meter altitude
    print("Flying to 1 meter altitude...")
    drone.move_up(100)
    time.sleep(2)

    running = True
    try:
        # Create a thread for moving to waypoints

        def waypoint_thread():
            print("Following waypoints...")
            move_to_waypoint(drone, wp)

        # Start the waypoint thread
        wp_thread = threading.Thread(target=waypoint_thread)
        wp_thread.start()

        # Infinite loop to display the video feed
        while running:
            # Capture the video frame
            
            frame = drone.get_frame_read().frame

            try:
                # Debug: Check if the frame is valid
                if frame is not None and frame.size > 0:
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
                            pil_image = torch.tensor(pil_image, dtype=torch.float).to(device)
                            pil_image = pil_image.unsqueeze(0)
                            
                            outputs = model(pil_image)
                            _, preds = torch.max(outputs.data, 1)

                        print("Swimming status:", lb.classes_[preds])
                        isDrowning = lb.classes_[preds] == 'drowning'

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

                        isDrowning = len(distances) > 0 and min(distances) < 50

                        out = draw_bbox(frame, bbox, label, conf, isDrowning)

                    else:
                        out = frame

                    batterytext = "Battery: {}%".format(drone.get_battery())
                    cv2.putText(out, batterytext, (5, 720 - 5), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)

                    # Convert the frame to a Pygame surface
                    frame_surface = frame_to_surface(out)

                    # Render the frame on the Pygame window
                    screen.blit(pygame.transform.scale(frame_surface, (screen.get_width(), screen.get_height())),(0, 0))
                    pygame.display.flip()
                else:
                    print("Warning: Frame is None or empty!")
            except Exception as e:
                print(f"Error processing frame: {e}")

            # Event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Add a small delay to limit CPU usage
            pygame.time.delay(10)

        # Wait for the waypoint thread to finish
        wp_thread.join()

    except KeyboardInterrupt:
        print("Manual interruption. Landing the drone...")

    finally:
        # Stop video stream and land the drone
        drone.streamoff()
        print("Landing...")
        drone.land()
        pygame.quit()

if __name__ == "__main__":
    main()
