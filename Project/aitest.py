from ultralytics import YOLO
import cv2
import asyncio
import websockets
import base64
import json
import numpy as np
import socket

# Load a pretrained YOLO model
model = YOLO("drowning.pt")

# Define path to video file
source = "video1.mp4"

# Get local IP address
def get_local_ip():
    try:
        # Create a socket to connect to an external server
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Doesn't actually connect but gets local routing info
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception:
        return "127.0.0.1"  # Default to localhost if cannot determine

# Process frame for websocket transmission
async def process_frame(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
    return jpg_as_text

# Handle websocket connections
async def handle_client(websocket):
    """Handle incoming websocket connections"""
    print(f"Client connected from {websocket.remote_address}")
    try:
        # Process video with YOLO model and stream frames
        results = model(source, stream=True)
        
        for r in results:
            # Get frame with detection annotations
            frame = r.plot()
            
            # Extract drowning detection information
            drowning_detected = False
            drowning_boxes = []
            
            if len(r.boxes) > 0:
                for box in r.boxes:
                    # Get class id and coordinates
                    cls_id = int(box.cls.item())
                    
                    # Assuming class 0 is the drowning class (adjust based on your model)
                    if cls_id == 0:  
                        drowning_detected = True
                        # Get the coordinates (convert to int for JSON serialization)
                        x1, y1, x2, y2 = [int(x) for x in box.xyxy[0].tolist()]
                        drowning_boxes.append({
                            "x1": x1, 
                            "y1": y1, 
                            "x2": x2, 
                            "y2": y2,
                            "center_x": int((x1 + x2) / 2),
                            "center_y": int((y1 + y2) / 2)
                        })
            
            # Convert to base64 for transmission
            encoded_frame = await process_frame(frame)
            
            # Send frame and detection info to client
            message = {
                "image": encoded_frame,
                "drowning_detected": drowning_detected,
                "drowning_boxes": drowning_boxes
            }
            
            await websocket.send(json.dumps(message))
            
            # Control frame rate
            await asyncio.sleep(0.03)  # ~30 FPS
            
    except websockets.exceptions.ConnectionClosed:
        print(f"Client disconnected: {websocket.remote_address}")

async def main():
    # Get the local IP address
    local_ip = get_local_ip()
    port = 8765
    
    # Start websocket server
    server = await websockets.serve(handle_client, "0.0.0.0", port)
    
    print(f"WebSocket server running at:")
    print(f"  • Local:   ws://localhost:{port}")
    print(f"  • Network: ws://{local_ip}:{port}")
    print("\nShare the Network URL with devices on your local network.")
    
    # Keep server running
    await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())
