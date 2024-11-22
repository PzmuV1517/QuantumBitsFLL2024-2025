import json
import time
import cv2
from djitellopy import Tello

tello= Tello()
tello.connect()

tello.streamon()
frame_read = tello.get_frame_read()
time.sleep(5)
cv2.imwrite("groupphoto.png", frame_read.frame)
time.sleep(5)
tello.land()