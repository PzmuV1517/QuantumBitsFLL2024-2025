import albumentations
import cv2
import numpy as np
from ultralytics import YOLO
import cvlib as cv
from cvlib.object_detection import draw_bbox

# Load YOLO model
print('Loading YOLO model...')
model = YOLO('best.pt')  # replace with your model path

aug = albumentations.Compose([
    albumentations.Resize(224, 224),
])

def detectDrowning():
    isDrowning = False

    # Use camera feed instead of video file
    cap = cv2.VideoCapture(0)  # Use camera index 0 (default camera)
    #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    if not cap.isOpened():
        print('Error: Unable to access the camera.')
        return

    while cap.isOpened():
        status, frame = cap.read()

        if not status:
            print('Error: Unable to capture video frame.')
            break

        # Apply object detection using YOLO
        results = model(frame)
        if len(results) > 0:
            result = results[0]
            bbox = result.boxes.xyxy.cpu().numpy().astype(int)  # Bounding boxes
            conf = result.boxes.conf.cpu().numpy()  # Confidence scores
            label = [model.names[int(cls)] for cls in result.boxes.cls.cpu().numpy()]  # Class labels

            # Check for the presence of the logo
            if len(bbox) > 0 and label[0] == 'logo-D7hc':
                isDrowning = True if conf[0] < 0.5 else False  # Example logic based on confidence

                # Change the label to "Logo" instead of "Normal"
                label = ["Logo" if lbl == 'logo-D7hc' else lbl for lbl in label]

                # Draw bounding box and label on the frame
                out = draw_bbox(frame, bbox, label, conf, isDrowning)
            else:
                out = frame
        else:
            out = frame

        # Display the output frame
        cv2.imshow("Real-time Drowning Detection", out)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()

# Start drowning detection using the camera feed
detectDrowning()
