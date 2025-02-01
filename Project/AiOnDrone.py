import json
import math
from djitellopy import Tello
import pygame
import cv2
import time
import threading
import numpy as np
from ultralytics import YOLO
import albumentations
import cvlib as cv
from cvlib.object_detection import draw_bbox

# Initialize Pygame
pygame.init()

# Constants for Pygame window
WINDOW_WIDTH = 960
WINDOW_HEIGHT = 720
WINDOW_TITLE = "Drone Camera Feed"

# Load YOLO model
print('Loading YOLO model...')
model = YOLO('best.pt')  # replace with your model path

aug = albumentations.Compose([
    albumentations.Resize(224, 224),
])

# Function to convert OpenCV frame to Pygame surface
def frame_to_surface(frame):
    # Convert BGR (OpenCV format) to RGB (Pygame format)
    #frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Transpose the frame to match Pygame's (width, height) format
    surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
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

    running = True
    try:
        # Infinite loop to display the video feed
        while running:
            # Capture the video frame
            frame = drone.get_frame_read().frame

            try:
                # Debug: Check if the frame is valid
                if frame is not None and frame.size > 0:
                    # Apply object detection using YOLO
                    results = model(frame)
                    if len(results) > 0:
                        result = results[0]
                        bbox = result.boxes.xyxy.cpu().numpy().astype(int)  # Bounding boxes
                        conf = result.boxes.conf.cpu().numpy()  # Confidence scores
                        label = [model.names[int(cls)] for cls in result.boxes.cls.cpu().numpy()]  # Class labels

                        # Check for the presence of the logo
                        if len(bbox) > 0 and label[0] == 'logo-D7hc':
                            isDrowning = True if conf[0] < 0.5 else False  # Example logic based on confidence

                            # Change the label to "Logo" instead of "Normal"
                            label = ["Logo" if lbl == 'logo-D7hc' else lbl for lbl in label]

                            # Draw bounding box and label on the frame
                            out = draw_bbox(frame, bbox, label, conf, isDrowning)
                        else:
                            out = frame
                    else:
                        out = frame

                    batterytext = "Battery: {}%".format(drone.get_battery())
                    cv2.putText(out, batterytext, (5, 720 - 5), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)

                    # Convert the frame to a Pygame surface
                    frame_surface = frame_to_surface(out)

                    # Render the frame on the Pygame window
                    screen.blit(pygame.transform.scale(frame_surface, (screen.get_width(), screen.get_height())), (0, 0))
                    pygame.display.flip()
                else:
                    print("Warning: Frame is None or empty!")
            except Exception as e:
                print(f"Error processing frame: {e}")

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