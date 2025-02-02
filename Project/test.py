from djitellopy import Tello
import keyboard
import time



def main():
    # Initialize drone
    drone = Tello()
    drone.connect()
    
    # Check battery
    battery = drone.get_battery()
    print(f"Battery: {battery}%")
    
    try:
        # Takeoff
        print("Taking off...")
        drone.takeoff()
        time.sleep(2)
        
        # Move forward 150cm
        print("Moving forward 150cm...")
        drone.move_forward(150)
        print("test")
        # Monitor for 'q' press
        while True:
            if keyboard.is_pressed('q'):
                print("Emergency stop activated!")
                drone.send_rc_control(0, 0, 0, 0)
                time.sleep(0.5)
                break
            time.sleep(0.1)
            
    except Exception as e:
        print(f"Error occurred: {e}")
        
    finally:
        # Safety landing
        print("Landing...")
        drone.land()
        drone.end()

if __name__ == "__main__":
    main()