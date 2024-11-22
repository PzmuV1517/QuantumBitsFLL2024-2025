from djitellopy import Tello

tello = Tello()
tello.connect()

print("Battery : " + str(tello.get_battery()))
print("Battery : " + str(tello.get_current_state()))