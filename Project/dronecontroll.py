import json
import math
from djitellopy import Tello
import time

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

# Main program
def main():
    # Load the JSON file
    with open("waypoint.json", "r") as file:
        data = json.load(file)

    wp = data['wp']  # Waypoints
    pos = data['pos']  # Positions (not used directly here)

    # Connect to Tello drone
    drone = Tello()
    drone.connect()

    # Check battery
    battery = drone.get_battery()
    print(f"Battery: {battery}%")
    if battery < 20:
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

    # Move along the waypoints
    print("Following waypoints...")
    move_to_waypoint(drone, wp)

    # Land the drone
    print("Landing...")
    drone.land()

if __name__ == "__main__":
    main()
