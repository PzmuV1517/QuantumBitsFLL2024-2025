import json
import time
import threading
import pygame
import cv2
from djitellopy import Tello
from ultralytics import YOLO

# Initialize Pygame
pygame.init()

# Constants for Pygame window
WINDOW_WIDTH = 960
WINDOW_HEIGHT = 720
WINDOW_TITLE = "Drone Camera Feed"

# Load YOLO model
print('Loading YOLO model...')
model = YOLO('best.pt')  # replace with your model path

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
    # Convert BGR (OpenCV format) to RGB (Pygame format)
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

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
