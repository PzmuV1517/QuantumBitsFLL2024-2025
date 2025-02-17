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
model = YOLO('logo.pt')  # replace with your model path

# Add these constants at the top with other constants
LOGO_DETECTION_THRESHOLD = 10  # Number of consecutive frames logo needs to be detected
DETECTION_RESET_THRESHOLD = 5  # Number of frames without detection before resetting counter

# Function to convert OpenCV frame to Pygame surface
def frame_to_surface(frame):

    # Transpose the frame to match Pygame's (width, height) format
    surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
    return surface

# Function to center the drone over the detected logo
def center_drone(drone, x_center, y_center, frame_width, frame_height):
    x_error = x_center - frame_width // 2
    y_error = y_center - frame_height // 2

    print(f"x_error: {x_error}, y_error: {y_error}")

    if abs(x_error) > 20:
        if x_error > 0:
            print("Moving right")
            drone.move_right(20)
            time.sleep(2)  # Add delay after movement
        else:
            print("Moving left")
            drone.move_left(20)
            time.sleep(2)  # Add delay after movement

    if abs(y_error) > 20:
        if y_error > 0:
            print("Moving back")
            drone.move_back(20)
            time.sleep(2)  # Add delay after movement
        else:
            print("Moving forward")
            drone.move_forward(20)
            time.sleep(2)  # Add delay after movement

# Function to move drone based on calculated distances and angles
def move_to_waypoint(drone, wp, stop_event):
    for point in wp:
        if stop_event.is_set():
            print("Waypoint function terminated.")
            time.sleep(2)  # Allow drone to stabilize
            return

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
    # Update battery threshold for flip maneuver
    if battery < 60:  # Increased from 40 to 60 since flip requires more power
        print("Battery too low for flip maneuver. Please charge the drone.")
        drone.streamoff()
        pygame.quit()
        return

    # Take off
    print("Taking off...")
    drone.takeoff()
    time.sleep(5)

    # Fly to 1 meter altitude
    print("Flying to 1 meter altitude...")
    drone.move_up(70)
    time.sleep(2)

    stop_event = threading.Event()
    centering_mode = False
    logo_detection_counter = 0
    frames_without_detection = 0

    running = True
    try:
        # Create a thread for moving to waypoints
        def waypoint_thread():
            print("Following waypoints...")
            move_to_waypoint(drone, wp, stop_event)

        # Start the waypoint thread
        wp_thread = threading.Thread(target=waypoint_thread)
        wp_thread.start()

        while running:
            # Capture the video frame
            frame = drone.get_frame_read().frame
            frame_height, frame_width, _ = frame.shape

            # Run the frame through the YOLO model
            results = model(frame)

            # Draw bounding boxes on the frame and check for logo
            logo_detected = False
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Check if the detected object is the logo
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    if class_name == 'logo-D7hc':  # Replace 'logo-D7hc' with the actual class name for the logo
                        logo_detected = True
                        x_center = (x1 + x2) // 2
                        y_center = (y1 + y2) // 2
                        print(f"Logo detected at: x_center={x_center}, y_center={y_center}")
                        
                        # Increment detection counter
                        logo_detection_counter += 1
                        frames_without_detection = 0
                        print(f"Logo detection counter: {logo_detection_counter}")
                        
                        # After 20 frames of detection, do a flip and land
                        if logo_detection_counter >= 10 and not stop_event.is_set():
                            print("Logo consistently detected for 10 frames. Stopping waypoints...")
                            stop_event.set()
                            time.sleep(2)  # Wait for drone to stabilize
                            
                            print("Performing flip...")
                            try:
                                # Move up a bit for safety
                                drone.move_up(30)
                                time.sleep(1)
                                # Check battery again before flip
                                if drone.get_battery() >= 50:
                                    drone.flip_forward()
                                    time.sleep(3)  # Wait for flip to complete
                                else:
                                    print("Battery too low for flip, landing directly")
                            except Exception as e:
                                print(f"Error during flip: {e}")
                            
                            print("Landing after flip...")
                            drone.land()
                            running = False

            # Reset counter if logo is not detected
            if not logo_detected:
                frames_without_detection += 1
                if frames_without_detection >= DETECTION_RESET_THRESHOLD:
                    if logo_detection_counter > 0:
                        print(f"Lost logo detection. Resetting counter. Previous count: {logo_detection_counter}")
                    logo_detection_counter = 0
                    frames_without_detection = 0

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

        centering_mode = True
        print("Starting centering mode...")

        # Start centering process after waypoint thread finishes
        while running:
            # Capture the video frame
            frame = drone.get_frame_read().frame
            frame_height, frame_width, _ = frame.shape

            # Run the frame through the YOLO model
            results = model(frame)

            # Draw bounding boxes on the frame and check for logo
            logo_detected = False
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Check if the detected object is the logo
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    if class_name == 'logo-D7hc':  # Replace 'logo-D7hc' with the actual class name for the logo
                        logo_detected = True
                        x_center = (x1 + x2) // 2
                        y_center = (y1 + y2) // 2
                        print(f"Logo detected at: x_center={x_center}, y_center={y_center}")
                        if centering_mode and logo_detected:
                            center_drone(drone, x_center, y_center, frame_width, frame_height)

            # Convert frame to Pygame surface
            surface = frame_to_surface(frame)

            # Display the frame
            screen.blit(surface, (0, 0))
            pygame.display.update()

            # Check for quit events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # If logo is detected and drone is centered, land the drone
            if logo_detected and abs(x_center - frame_width // 2) <= 20 and abs(y_center - frame_height // 2) <= 20:
                print("Logo detected and centered. Landing the drone...")
                drone.land()
                running = False

            # Add a small delay to limit CPU usage
            pygame.time.delay(10)

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