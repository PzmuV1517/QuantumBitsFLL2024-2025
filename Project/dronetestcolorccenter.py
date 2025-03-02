import cv2
import pygame
from djitellopy import Tello
from ultralytics import YOLO
import time
import numpy as np
from pygame import mixer
import threading

WINDOW_TITLE = "Drone Camera Feed"
FORWARD_SPEED = 15  # Speed setting (10-100)
CENTER_THRESHOLD = 60  # Pixel threshold for considering drone centered (adjust as needed)
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
MAX_CENTERING_ATTEMPTS = 5  # Maximum number of centering attempts
CENTERING_SPEED_FACTOR = 0.08  # Reduced from 0.1 for slower, more precise movements

# Load YOLO model
print('Loading YOLO model...')
model = YOLO('logoColor.pt')  # replace with your model path

# Initialize Pygame mixer for audio
pygame.mixer.init()

# Audio playback control
audio_playing = False
audio_thread = None
audio_stop_event = threading.Event()  # Event to signal audio thread to stop

def play_audio_loop(audio_file):
    """Function to play audio in a loop until stopped"""
    global audio_playing
    audio_playing = True
    try:
        pygame.mixer.music.load(audio_file)
        while not audio_stop_event.is_set():
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy() and not audio_stop_event.is_set():
                time.sleep(0.1)
    except Exception as e:
        print(f"Audio playback error: {e}")
    audio_playing = False

def start_audio(audio_file):
    """Start playing the specified audio file in a separate thread"""
    global audio_thread, audio_stop_event
    # Stop any currently playing audio
    stop_audio()
    
    audio_stop_event.clear()
    try:
        audio_thread = threading.Thread(target=play_audio_loop, args=(audio_file,))
        audio_thread.daemon = True
        audio_thread.start()
        print(f"Audio playback started: {audio_file}")
    except Exception as e:
        print(f"Error starting audio: {e}")

def stop_audio():
    """Stop the audio playback"""
    global audio_thread, audio_stop_event
    if audio_thread is not None and audio_thread.is_alive():
        audio_stop_event.set()
        pygame.mixer.music.stop()
        audio_thread.join(timeout=1.0)
        print("Audio playback stopped")

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
    left_right = int(-dx * CENTERING_SPEED_FACTOR)  # Reduced speed factor for more precise movements
    forward_backward = int(-dy * CENTERING_SPEED_FACTOR)  # Reduced speed factor for more precise movements
    
    # Limit movement commands to avoid overcompensation
    left_right = max(-20, min(20, left_right))  # Reduced from ±30 to ±20
    forward_backward = max(-20, min(20, forward_backward))  # Reduced from ±30 to ±20
    
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
    frame_center = (FRAME_WIDTH // 2 + 60 , FRAME_HEIGHT - 40)
    is_centering = False
    is_landing = False

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

            # Handle landing status if needed
            if is_landing:
                # Nothing to do here, just continue displaying frames
                pass
            # Handle logo detection and centering
            elif is_centering:
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
                        
                        # Set landing flag and start landing audio
                        is_landing = True
                        is_centering = False  # No longer centering
                        
                        # Stop centering audio and start landing audio
                        stop_audio()
                        start_audio("landing.mp3")
                        
                        # Land the drone
                        print("Landing sequence initiated...")
                        drone.land()
                        time.sleep(1)
                        # Don't break here - continue loop to show frames and play audio
                    else:
                        # Move to center over the logo - slower movements
                        print(f"Centering: LR {left_right}, FB {forward_backward}")
                        drone.send_rc_control(-left_right, -forward_backward, 0, 0)
                        time.sleep(0.8)  # Increased from 0.5 to 0.8 for slower movements
                        drone.send_rc_control(0, 0, 0, 0)  # Stop briefly to stabilize
                        time.sleep(0.7)  # Increased from 0.5 to 0.7 for more stability time
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
                        
                        # Start playing centering audio
                        start_audio("centering.mp3")
                        
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
            try:
                surface = frame_to_surface(frame)
                screen.blit(surface, (0, 0))
                pygame.display.update()
            except Exception as e:
                print(f"Display error: {e}")

            # Check if drone has landed and we should exit
            if is_landing and not drone.is_flying:
                # Keep displaying frames for a bit after landing
                # to show the landing visuals and keep playing audio
                time.sleep(3)  # Show landing for 3 seconds
                break

            # Add a small delay to prevent CPU overuse
            time.sleep(0.03)  # 30fps approximately

            # Check for quit events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    stop_audio()
                    if not is_landing:
                        drone.land()
                    return

    finally:
        # Clean up
        stop_audio()
        drone.send_rc_control(0, 0, 0, 0)  # Ensure drone stops
        drone.streamoff()
        pygame.quit()
        if drone.is_flying:
            drone.land()

if __name__ == "__main__":
    main()