#!/usr/bin/env python3

"""
Runs Hailo detection using PiCamera2 and streams annotated video
and detection data via WebSocket only when detections are found.
"""

import argparse
import asyncio
import base64
import json
import socket
import cv2
import websockets
from picamera2 import MappedArray, Picamera2
from picamera2.devices import Hailo
import os # Import os to construct relative paths

# --- Global Variables ---
# Store connected WebSocket clients
connected_clients = set()
# Shared data between detection loop and websockets
latest_data = {
    "image": None,
    "detections": [],
    "detection_count": 0
}
# Lock for safe access to shared data
data_lock = asyncio.Lock()
# --- Helper Functions ---

def get_local_ip():
    """Gets the local IP address of the machine."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception:
        return "127.0.0.1" # Default to localhost if unable to determine

def extract_detections(hailo_output, w, h, class_names, threshold=0.5):
    """
    Extracts detections from the HailoRT-postprocess output.
    Returns a list of [class_name, bbox, score, center_x, center_y].
    """
    results = []
    for class_id, detections in enumerate(hailo_output):
        for detection in detections:
            score = detection[4]
            if score >= threshold:
                y0, x0, y1, x1 = detection[:4]
                # Scale bounding box to original frame dimensions
                abs_x0 = int(x0 * w)
                abs_y0 = int(y0 * h)
                abs_x1 = int(x1 * w)
                abs_y1 = int(y1 * h)
                bbox = (abs_x0, abs_y0, abs_x1, abs_y1)
                center_x = int((abs_x0 + abs_x1) / 2)
                center_y = int((abs_y0 + abs_y1) / 2)
                # Ensure class_id is within bounds of class_names
                if 0 <= class_id < len(class_names):
                    results.append([class_names[class_id], bbox, score, center_x, center_y])
                else:
                    print(f"Warning: Detected class ID {class_id} out of bounds for labels file.")
                    results.append(["unknown", bbox, score, center_x, center_y]) # Handle unknown class
    return results

def draw_detections_on_frame(frame, current_detections):
    """Draws bounding boxes and labels directly onto a frame."""
    if current_detections:
        for class_name, bbox, score, _, _ in current_detections:
            x0, y0, x1, y1 = bbox
            label = f"{class_name} {int(score * 100)}%"
            # Draw rectangle (Green)
            cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
            # Put label background
            (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x0, y0 - label_height - baseline), (x0 + label_width, y0), (0, 255, 0), cv2.FILLED)
            # Put label text (Black)
            cv2.putText(frame, label, (x0, y0 - baseline // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return frame

async def broadcast_data():
    """Sends the latest data to all connected clients."""
    async with data_lock:
        # Only broadcast if there are detections AND connected clients
        if latest_data["detection_count"] > 0 and connected_clients:
            # Prepare JSON data
            message_data = {
                "image": latest_data["image"],
                "detections": [ # Format detection data for JSON
                    {
                        "class_name": det[0],
                        "bbox": det[1], # (x0, y0, x1, y1)
                        "score": float(det[2]), # Ensure score is float
                        "center_x": det[3],
                        "center_y": det[4]
                    } for det in latest_data["detections"]
                ],
                "detection_count": latest_data["detection_count"]
            }
            message = json.dumps(message_data)
            # Use asyncio.gather to send messages concurrently
            tasks = [client.send(message) for client in connected_clients]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            # Handle potential errors during send (e.g., client disconnected)
            disconnected_clients = set()
            for i, result in enumerate(results):
                 if isinstance(result, Exception):
                    client = list(connected_clients)[i] # Get corresponding client
                    print(f"Error sending to client {client.remote_address}: {result}. Removing client.")
                    disconnected_clients.add(client) # Mark client for removal

            # Remove disconnected clients outside the loop
            for client in disconnected_clients:
                connected_clients.discard(client)

        # Clear image data after attempting broadcast to save memory,
        # but keep detections until next update cycle finds new ones (or none)
        latest_data["image"] = None

async def handle_client(websocket):
    """Handles a single WebSocket client connection."""
    print(f"Client connected: {websocket.remote_address}")
    connected_clients.add(websocket)
    try:
        # Keep the connection alive, data is pushed by broadcast_data
        await websocket.wait_closed()
    except websockets.exceptions.ConnectionClosedOK:
        print(f"Client disconnected normally: {websocket.remote_address}")
    except websockets.exceptions.ConnectionClosedError as e:
        print(f"Client connection closed with error: {websocket.remote_address} - {e}")
    finally:
        print(f"Removing client: {websocket.remote_address}")
        connected_clients.discard(websocket) # Use discard to avoid error if already removed

async def run_detection_loop(args):
    """Initializes camera/Hailo and runs the main detection loop."""
    global latest_data

    # Construct absolute path for model and labels relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, args.model)
    labels_path = os.path.join(script_dir, args.labels)

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    if not os.path.exists(labels_path):
        print(f"Error: Labels file not found at {labels_path}")
        return

    # Get the Hailo model, the input size it wants, and the size of our preview stream.
    try:
        with Hailo(model_path) as hailo:
            model_h, model_w, _ = hailo.get_input_shape()
            video_w, video_h = 1280, 960 # High-res stream size

            # Load class names from the labels file
            try:
                with open(labels_path, 'r', encoding="utf-8") as f:
                    class_names = f.read().splitlines()
            except Exception as e:
                print(f"Error reading labels file {labels_path}: {e}")
                return

            # Configure and start Picamera2.
            with Picamera2() as picam2:
                main_stream_config = {'size': (video_w, video_h), 'format': 'RGB888'} # Use RGB for cv2 compatibility
                lores_stream_config = {'size': (model_w, model_h), 'format': 'RGB888'}
                controls = {'FrameRate': 30}
                config = picam2.create_preview_configuration(
                    main=main_stream_config,
                    lores=lores_stream_config,
                    controls=controls,
                    encode="lores" # Encode the lores stream for capture_array
                )
                picam2.configure(config)
                picam2.start()
                print("Picamera2 started successfully.")
                print(f"Using Hailo model: {model_path}")
                print(f"Using labels: {labels_path}")
                print(f"Detection threshold: {args.score_thresh}")

                # Process frames continuously
                while True:
                    # Capture high-res frame for display/annotation and low-res for inference
                    main_frame = picam2.capture_array('main')
                    lores_frame = picam2.capture_array('lores')

                    # Run inference on the low-resolution frame
                    hailo_output = hailo.run(lores_frame)

                    # Extract detections, scaling bboxes to the main frame size
                    current_detections = extract_detections(
                        hailo_output, video_w, video_h, class_names, args.score_thresh
                    )

                    detection_count = len(current_detections)

                    # Update shared data safely
                    async with data_lock:
                        latest_data["detections"] = current_detections
                        latest_data["detection_count"] = detection_count
                        # Only encode and store image if detections were found
                        if detection_count > 0:
                            # Annotate the high-resolution frame
                            annotated_frame = draw_detections_on_frame(main_frame.copy(), current_detections)
                            # Encode the annotated frame to JPEG and then Base64
                            _, buffer = cv2.imencode('.jpg', annotated_frame)
                            jpg_as_text = base64.b64encode(buffer).decode('utf-8')
                            latest_data["image"] = jpg_as_text
                        else:
                            # Ensure image is None if no detections
                            latest_data["image"] = None

                    # Schedule broadcast task (don't wait for it here)
                    # broadcast_data itself checks if detection_count > 0 before sending
                    asyncio.create_task(broadcast_data())

                    # Yield control to the asyncio event loop briefly
                    await asyncio.sleep(0.01) # Adjust sleep time as needed

    except Exception as e:
        print(f"Error during Hailo/Picamera2 initialization or loop: {e}")
        # Consider more specific error handling if needed


async def main():
    # Parse command-line arguments.
    parser = argparse.ArgumentParser(description="Hailo Detection WebSocket Server")
    # Default model path assumes it's in the same directory as the script
    parser.add_argument("-m", "--model", help="Path for the HEF model (relative to script).",
                        default="yolov8s_h8l.hef")
    # Default labels path assumes it's in the same directory
    parser.add_argument("-l", "--labels", default="coco.txt",
                        help="Path to a text file containing labels (relative to script).")
    parser.add_argument("-s", "--score_thresh", type=float, default=0.5,
                        help="Score threshold for detections (0.0 to 1.0).")
    parser.add_argument("--port", type=int, default=8765,
                        help="WebSocket server port.")
    args = parser.parse_args()

    # Validate score threshold
    if not 0.0 <= args.score_thresh <= 1.0:
        print("Error: Score threshold must be between 0.0 and 1.0.")
        return

    local_ip = get_local_ip()
    port = args.port

    # Start the WebSocket server
    server = await websockets.serve(handle_client, "0.0.0.0", port)
    print(f"WebSocket server starting at:")
    print(f"  • Local:   ws://localhost:{port}")
    print(f"  • Network: ws://{local_ip}:{port}")
    print("\nWaiting for connections...")
    print("Detection loop starting. Will broadcast data when detections occur.")

    # Run the detection loop concurrently
    detection_task = asyncio.create_task(run_detection_loop(args))

    # Keep the server running until interrupted
    try:
        await asyncio.gather(server.wait_closed(), detection_task)
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.close()
        await server.wait_closed()
        detection_task.cancel()
        try:
            await detection_task
        except asyncio.CancelledError:
            print("Detection loop cancelled.")
    except Exception as e:
        print(f"An unexpected error occurred in main: {e}")
        # Clean up tasks if they are running
        if not detection_task.done():
            detection_task.cancel()
        if server.is_serving():
             server.close()
             await server.wait_closed()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except RuntimeError as e:
        print(f"Runtime Error: {e}. Ensure Hailo drivers and runtime are correctly installed and configured.")
    except ImportError as e:
         print(f"Import Error: {e}. Make sure all required libraries (picamera2, hailo-hailort, websockets, opencv-python) are installed.")
