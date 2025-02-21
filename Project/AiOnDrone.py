import cv2
import pygame
from djitellopy import Tello
from ultralytics import YOLO
import albumentations

WINDOW_TITLE = "Drone Camera Feed"

# Load YOLO model
print('Loading YOLO model...')
model = YOLO('logoColor.pt')  # replace with your model path

# Function to convert OpenCV frame to Pygame surface
def frame_to_surface(frame):
    # Convert BGR (OpenCV format) to RGB (Pygame format)
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Transpose the frame to match Pygame's (width, height) format
    surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
    return surface

# Main program
def main():
    drone = Tello()
    drone.connect()

    # Start the video stream
    drone.streamon()

    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((960, 720))
    pygame.display.set_caption(WINDOW_TITLE)

    try:
        while True:
            # Get the frame from the drone
            frame = drone.get_frame_read().frame


            # Run the frame through the YOLO model
            results = model(frame)

            # Draw bounding boxes on the frame
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Convert frame to Pygame surface
            surface = frame_to_surface(frame)

            # Display the frame
            screen.blit(surface, (0, 0))
            pygame.display.update()

            # Check for quit events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return

    finally:
        # Clean up
        drone.streamoff()
        pygame.quit()

if __name__ == "__main__":
    main()