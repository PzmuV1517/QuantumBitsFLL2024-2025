from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("best.pt")

# Define path to video file
source = "video1.mp4"

# Run inference on the source
results = model(source, show=True)  # generator of Results objects