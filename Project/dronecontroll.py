import json
from djitellopy import Tello

# Load the JSON file
with open('waypoint.json', 'r') as file:
    data = json.load(file)

# Extract waypoints
waypoints = data['wp']

# Initialize Tello drone
tello = Tello()
tello.connect()

# Get battery status
print(f"Battery: {tello.get_battery()}%")

# Takeoff
tello.takeoff()

# Execute the path based on waypoints
for i, wp in enumerate(waypoints):
    try:
        distance = wp['dist_cm']
        angle = wp['angle_deg']
        print(f"Battery: {tello.get_battery()}%")
        
        # Move forward the specified distance
        print(f"Waypoint {i+1}: Moving forward {distance} cm")
        tello.move_forward(distance)
        
        # Rotate clockwise by the specified angle
        print(f"Waypoint {i+1}: Rotating {angle} degrees")
        tello.rotate_clockwise(angle)
    except Exception as e:
        print(f"Error at waypoint {i+1}: {e}")

# Land the drone
print("Path completed. Landing...")
tello.land()
