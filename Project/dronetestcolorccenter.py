import cv2
import pygame
from djitellopy import Tello
from ultralytics import YOLO
import time
import numpy as np

WINDOW_TITLE = "Drone Camera Feed"
FORWARD_SPEED = 15  # Speed setting (10-100)
CENTER_THRESHOLD = 30  # Pixel threshold for considering drone centered (adjust as needed)
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
MAX_CENTERING_ATTEMPTS = 5  # Maximum number of centering attempts

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

# Function to calculate movement vector
def calculate_centering_vector(frame_center, logo_center):
    # Calculate distance in pixels from frame center to logo center
    dx = logo_center[0] - frame_center[0]  
    dy = logo_center[1] - frame_center[1]
    
    # Convert pixel distances to drone movement commands
    # Note: Forward/backward is reversed because camera is mirrored
    # Left/right is also reversed due to the mirror
    left_right = int(-dx * 0.1)  # Negative because mirror flips horizontally
    forward_backward = int(-dy * 0.1)  # Negative because mirror flips vertically
    
    # Limit movement commands to avoid overcompensation
    left_right = max(-30, min(30, left_right))
    forward_backward = max(-30, min(30, forward_backward))
    
    return left_right, forward_backward, abs(dx), abs(dy)

# Main program
def main():
    drone = Tello()
    drone.connect()

    # Battery Percentage
    print(f"Battery: {drone.get_battery()}%")

    # Start the video stream
    drone.streamon()

    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((FRAME_WIDTH, FRAME_HEIGHT))
    pygame.display.set_caption(WINDOW_TITLE)

    # Take off and rise to 1 meter
    print("Taking off...")
    drone.takeoff()
    time.sleep(2)
    drone.move_up(30)
    time.sleep(2)

    # Set forward speed
    drone.set_speed(FORWARD_SPEED)
    print(f"Moving forward at speed {FORWARD_SPEED}...")

    # Initialize variables for logo detection buffer
    logo_detected_frames = 0
    logo_positions = []
    frame_center = (FRAME_WIDTH // 2 - 10, FRAME_HEIGHT - 40)
    is_centering = False

    try:
        while True:
            # Get the frame from the drone
            frame = drone.get_frame_read().frame

            # Run the frame through the YOLO model
            results = model(frame)

            # Draw bounding boxes on the frame
            logo_detected = False
            logo_center = None
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    if class_name == 'QuantumBits' and confidence > 0.7:
                        logo_detected = True
                        # Calculate the center of the logo
                        logo_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                        # Draw center point of logo
                        cv2.circle(frame, logo_center, 5, (0, 0, 255), -1)
                        # Draw confidence score on frame
                        cv2.putText(frame, f"C: {confidence:.3f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Draw the frame center for reference
            cv2.circle(frame, frame_center, 5, (255, 0, 0), -1)
            cv2.line(frame, (frame_center[0]-20, frame_center[1]), (frame_center[0]+20, frame_center[1]), (255, 0, 0), 2)
            cv2.line(frame, (frame_center[0], frame_center[1]-20), (frame_center[0], frame_center[1]+20), (255, 0, 0), 2)

            # Handle logo detection and drone movement
            if is_centering:
                # We're in centering mode
                if logo_detected and logo_center:
                    left_right, forward_backward, dx_abs, dy_abs = calculate_centering_vector(frame_center, logo_center)
                    
                    # Display vector info on frame
                    cv2.putText(frame, f"LR: {left_right}, FB: {forward_backward}", (20, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                    # Draw vector line
                    cv2.line(frame, frame_center, logo_center, (0, 255, 255), 2)
                    
                    # Check if centered
                    if dx_abs <= CENTER_THRESHOLD and dy_abs <= CENTER_THRESHOLD:
                        print("Drone centered over logo!")
                        drone.send_rc_control(0, 0, 0, 0)  # Stop all movement
                        time.sleep(1)
                        
                        # Land the drone
                        print("Landing sequence initiated...")
                        drone.land()
                        time.sleep(1)
                        break
                    else:
                        # Move to center over the logo
                        print(f"Centering: LR {left_right}, FB {forward_backward}")
                        drone.send_rc_control(-left_right, -forward_backward, 0, 0)
                        time.sleep(0.5)
                        drone.send_rc_control(0, 0, 0, 0)  # Stop briefly to stabilize
                        time.sleep(0.5)
                else:
                    # Lost sight of the logo during centering
                    print("Logo lost during centering, hovering...")
                    drone.send_rc_control(0, 0, 0, 0)
                    
            else:
                # Normal detection mode
                if logo_detected:
                    logo_detected_frames += 1
                    if logo_center:
                        logo_positions.append(logo_center)
                    
                    # If detected for 3+ consecutive frames, start centering
                    if logo_detected_frames >= 3:
                        print("Logo confirmed for 3 frames - Starting centering...")
                        drone.send_rc_control(0, 0, 0, 0)  # Stop all movement
                        time.sleep(1)  # Stabilize
                        
                        # Use the last detected logo position for initial centering
                        is_centering = True
                        logo_detected_frames = 0
                        logo_positions = []
                    else:
                        # Slow down when logo is detected but not yet confirmed
                        drone.send_rc_control(0, FORWARD_SPEED // 2, 0, 0)
                else:
                    # Reset counter if logo detection lost
                    logo_detected_frames = 0
                    logo_positions = []
                    # Move forward if no logo detected
                    drone.send_rc_control(0, FORWARD_SPEED, 0, 0)

            # Convert and display frame
            surface = frame_to_surface(frame)
            screen.blit(surface, (0, 0))
            pygame.display.update()

            # Add a small delay to prevent CPU overuse
            time.sleep(0.03)  # 30fps approximately

            # Check for quit events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    drone.land()
                    return

    finally:
        # Clean up
        drone.send_rc_control(0, 0, 0, 0)  # Ensure drone stops
        drone.streamoff()
        pygame.quit()
        if drone.is_flying:
            drone.land()

if __name__ == "__main__":
    main()