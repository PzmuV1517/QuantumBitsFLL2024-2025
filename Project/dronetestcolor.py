import cv2
import pygame
from djitellopy import Tello
from ultralytics import YOLO
import time

WINDOW_TITLE = "Drone Camera Feed"
FORWARD_SPEED = 20  # Speed setting (10-100)

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

    # Take off and rise to 1 meter
    print("Taking off...")
    drone.takeoff()
    time.sleep(2)
    drone.move_up(30)
    time.sleep(2)

    # Set forward speed
    drone.set_speed(FORWARD_SPEED)
    print(f"Moving forward at speed {FORWARD_SPEED}...")

    try:
        while True:
            # Get the frame from the drone
            frame = drone.get_frame_read().frame

            # Run the frame through the YOLO model
            results = model(frame)

            # Draw bounding boxes on the frame
            logo_detected = False
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    if class_name == 'QuantumBits' and confidence > 0.8:  # Added confidence threshold
                        logo_detected = True
                        print(f"Logo detected with confidence: {confidence:.2f}")
                        # Draw confidence score on frame
                        cv2.putText(frame, f"Conf: {confidence:.2f}", (x1, y1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # If logo detected, stop and land
            if logo_detected:
                print("Logo confirmed - Stopping drone...")
                drone.send_rc_control(0, 0, 0, 0)  # Stop all movement
                time.sleep(2)  # Hover for 2 seconds
                
                print("Landing sequence initiated...")
                drone.land()
                time.sleep(1)  # Wait for landing to complete
                break  # Exit the loop after landing

            else:
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