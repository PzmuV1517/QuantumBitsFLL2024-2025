import json
import time
import threading
import pygame
import cv2
from djitellopy import Tello
from ultralytics import YOLO
from datetime import datetime

# Add debug function at top of file
def debug_detection(box):
    """Debug helper to print detection details"""
    class_id = int(box.cls[0])
    class_name = model.names[class_id]
    confidence = float(box.conf[0])
    print(f"Detection: class={class_name}, confidence={confidence:.2f}")
    log_action(f"Detection: class={class_name}, confidence={confidence:.2f}")

# Initialize Pygame
pygame.init()

# Constants for Pygame window
WINDOW_WIDTH = 960
WINDOW_HEIGHT = 720
WINDOW_TITLE = "Drone Camera Feed"
CENTERING_THRESHOLD = 20  # Pixels threshold for considering drone centered

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


    # Transpose the frame to match Pygame's (width, height) format
    surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
    return surface

# Function to log drone actions
def log_action(action):
    """Write drone actions to log file with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("drone_log.txt", "a") as f:
        f.write(f"[{timestamp}] {action}\n")

# Function to center the drone over the detected logo
def center_drone(drone, x_center, y_center, frame_width, frame_height):
    x_error = x_center - frame_width // 2
    y_error = y_center - frame_height // 2

    print(f"x_error: {x_error}, y_error: {y_error}")

    if abs(x_error) > CENTERING_THRESHOLD:
        if x_error > 0:
            log_action("Moving right")
            print("Moving right")
            drone.move_right(20)
        else:
            log_action("Moving left")
            print("Moving left")
            drone.move_left(20)
        return False

    if abs(y_error) > CENTERING_THRESHOLD:
        if y_error > 0:
            log_action("Moving back")
            print("Moving back")
            drone.move_back(20)
        else:
            log_action("Moving forward")
            print("Moving forward")
            drone.move_forward(20)
        return False
    
    return True  # Return True if drone is centered

def stop_all_movement(drone):
    """Emergency stop all drone movement"""
    drone.send_rc_control(0, 0, 0, 0)
    time.sleep(0.1)  # Short pause to ensure commands register
    log_action("Emergency stop - all movement halted")

# Main program
def main():
    log_action("Drone program started")
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
    drone.move_up(50)
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
            frame_height, frame_width, _ = frame.shape

            # Run the frame through the YOLO model
            results = model(frame)

            # Draw bounding boxes on the frame and check for logo
            logo_detected = False
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Debug each detection
                    debug_detection(box)
                    
                    # Get actual class name from model
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    
                    # Compare with actual class name
                    if class_name == 'logo-D7hc':  # Verify this matches your model's class name
                        if not logo_detected:
                            print("Logo detected - stopping movement")
                            stop_all_movement(drone)
                            wp_thread.join(timeout=1.0)  # Wait for waypoint thread to stop
                        
                        logo_detected = True
                        x_center = (x1 + x2) // 2
                        y_center = (y1 + y2) // 2
                        
                        if center_drone(drone, x_center, y_center, frame_width, frame_height):
                            print("Logo centered - Landing drone")
                            drone.land()
                            running = False
                            break

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
        log_action("Drone program ended")

if __name__ == "__main__":
    main()
