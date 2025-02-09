import time
from djitellopy import Tello

SPEED = 50


def custom_move(drone, direction, distance_cm):
    duration = distance_cm / SPEED  # Calculate duration based on speed and distance

    if direction == "left":
        drone.send_rc_control(-SPEED, 0, 0, 0)
    elif direction == "right":
        drone.send_rc_control(SPEED, 0, 0, 0)
    elif direction == "forward":
        drone.send_rc_control(0, SPEED, 0, 0)
    elif direction == "back":
        drone.send_rc_control(0, -SPEED, 0, 0)
    elif direction == "up":
        drone.send_rc_control(0, 0, SPEED, 0)
    elif direction == "down":
        drone.send_rc_control(0, 0, -SPEED, 0)

    time.sleep(duration)
    drone.send_rc_control(0, 0, 0, 0)


drone = Tello()
drone.connect()


drone.takeoff()
time.sleep(5)

custom_move(drone, "forward", 100)


