import time
import pygame
import cv2
from djitellopy import Tello
from ultralytics import YOLO

# Initialize Pygame
pygame.init()

# Constants
WINDOW_WIDTH = 960
WINDOW_HEIGHT = 720
WINDOW_TITLE = "Drone Camera Feed"
FORWARD_SPEED = 30  # Speed setting (10-100)
DETECTION_THRESHOLD = 5  # Number of frames to confirm logo detection

def frame_to_surface(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
    return surface

def main():
    # Load YOLO model
    print('Loading YOLO model...')
    model = YOLO('best.pt')

    # Initialize drone
    drone = Tello()
    drone.connect()
    
    # Start video stream
    drone.streamon()
    
    # Create display window
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption(WINDOW_TITLE)
    
    # Check battery
    battery = drone.get_battery()
    print(f"Battery: {battery}%")
    if battery < 40:
        print("Battery too low. Please charge the drone.")
        return
    
    # Take off and rise to 1 meter
    print("Taking off...")
    drone.takeoff()
    time.sleep(2)
    drone.move_up(70)
    time.sleep(2)
    
    # Set forward speed
    drone.set_speed(FORWARD_SPEED)
    
    running = True
    logo_detection_counter = 0
    
    try:
        # Start moving forward
        print(f"Moving forward at speed {FORWARD_SPEED}...")
        drone.send_rc_control(0, FORWARD_SPEED, 0, 0)  # Forward movement
        
        while running:
            # Get and process frame
            frame = drone.get_frame_read().frame
            results = model(frame)
            
            # Check for logo
            logo_detected = False
            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    if class_name == 'logo-D7hc':
                        logo_detected = True
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        print("Logo detected!")
                        logo_detection_counter += 1
                        break
            
            # If logo not detected, reset counter
            if not logo_detected:
                logo_detection_counter = 0
            
            # If logo detected consistently, stop and land
            if logo_detection_counter >= DETECTION_THRESHOLD:
                print("Logo confirmed! Stopping...")
                drone.send_rc_control(0, 0, 0, 0)  # Stop movement
                time.sleep(2)  # Hover for 2 seconds
                print("Landing...")
                drone.land()
                running = False
            
            # Display frame
            surface = frame_to_surface(frame)
            screen.blit(surface, (0, 0))
            pygame.display.flip()
            
            # Check for quit events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            # Small delay
            pygame.time.delay(10)
    
    except KeyboardInterrupt:
        print("Manual interruption...")
    
    finally:
        # Cleanup
        drone.send_rc_control(0, 0, 0, 0)  # Ensure drone stops
        drone.streamoff()
        pygame.quit()
        if drone.is_flying:
            drone.land()

if __name__ == "__main__":
    main()