import json
import time
import cv2
from djitellopy import Tello

tello= Tello()
tello.connect()

tello.streamon()
frame_read = tello.get_frame_read()


tello.takeoff()

flip_back()

cv2.imwrite("groupphoto.png", frame_read.frame)

tello.land()