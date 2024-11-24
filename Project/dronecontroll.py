import json
import time
from djitellopy import Tello

# Load JSON data
with open('waypoint.json', 'r') as file:
    data = json.load(file)

waypoints = data['wp']

# Initialize Tello
tello = Tello()
tello.connect()

# Speed controller (1-100 cm/s)
default_speed = 65  # Default speed
near_speed = 20     # Speed when nearing a waypoint
deceleration_threshold = 30  # Distance (cm) to start decelerating

# Set initial speed
tello.set_speed(default_speed)

# Get battery status
print(f"Battery: {tello.get_battery()}%")

# Takeoff
tello.takeoff()

# Execute waypoints
for i, wp in enumerate(waypoints):
    try:
        distance = wp['dist_cm']
        angle = wp['angle_deg']

        remaining_distance = distance
        step_size = 100  # Maximum step size (cm)

        print(f"Waypoint {i+1}: Moving {distance} cm in steps")
        while remaining_distance > 0:
            # Calculate next step size
            move_distance = min(step_size, remaining_distance)

            # Adjust speed when nearing the waypoint
            if remaining_distance <= deceleration_threshold:
                print(f"Decelerating to {near_speed} cm/s")
                tello.set_speed(near_speed)
            else:
                tello.set_speed(default_speed)

            # Move forward the calculated distance
            tello.move_forward(move_distance)
            remaining_distance -= move_distance
            print(f"Remaining distance: {remaining_distance} cm")
            time.sleep(1)  # Allow drone to stabilize

        # Reset to default speed after reaching waypoint
        tello.set_speed(default_speed)

        # Rotate (optimize direction)
        if angle != 0:
            print(f"Waypoint {i+1}: Rotating {angle} degrees")
            if angle > 0:
                tello.rotate_clockwise(angle)
            else:
                tello.rotate_counter_clockwise(abs(angle))
            time.sleep(1)  # Allow stabilization after rotation

    except Exception as e:
        print(f"Error at waypoint {i+1}: {e}")

# Land
print("Path completed. Landing...")
tello.land()
