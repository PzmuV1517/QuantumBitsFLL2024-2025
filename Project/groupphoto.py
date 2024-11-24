import time
import cv2
from djitellopy import Tello

# Initialize Tello and connect
tello = Tello()
tello.connect()
print(f"Battery Life: {tello.get_battery()}%")

# Start the video stream
tello.streamon()
frame_read = tello.get_frame_read()

# Takeoff
tello.takeoff()
time.sleep(2)  # Give the drone some time to stabilize
tello.move_up(30)
time.sleep(2) # Give the drone some time to stabilize
tello.flip_back()
time.sleep(2)

# Capture the frame
frame = frame_read.frame  # Get the current frame
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB format

# Save the image
cv2.imwrite("groupphoto.png", frame_rgb)
print("Image saved as 'groupphoto.png'")
time.sleep(2)

# Turn off the video stream
tello.streamoff()

# Wait before landing
tello.land()


