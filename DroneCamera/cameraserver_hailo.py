import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import asyncio
import websockets
import base64
import json
import numpy as np
import socket
import cv2
import threading
import hailo
import os

from hailo_apps_infra.hailo_rpi_common import (
    get_caps_from_pad,
    get_numpy_from_buffer,
)

# --- Configuration ---
HEF_FILE_PATH = "logov8s.hef" # Path to your HEF file
# Choose 'usb' for USB camera or 'rpi' for Raspberry Pi camera
CAMERA_TYPE = 'usb'
# CAMERA_TYPE = 'rpi'

# --- Global Variables ---
latest_frame = None
latest_detections = []
data_lock = threading.Lock()
gst_pipeline = None
main_loop = None

# --- Hailo/GStreamer Specific Functions ---

def get_source_element(camera_type):
    """Creates the appropriate GStreamer source element."""
    if camera_type == 'usb':
        # Using V4L2 for USB cameras
        src_element = Gst.ElementFactory.make("v4l2src", "source")
        # Optional: Specify device if not /dev/video0
        # src_element.set_property("device", "/dev/videoX")
    elif camera_type == 'rpi':
        # Using libcamera for Raspberry Pi cameras
        src_element = Gst.ElementFactory.make("libcamerasrc", "source")
        src_element.set_property("auto-focus-mode", 2) # Continuous focus
    else:
        raise ValueError("Invalid CAMERA_TYPE. Choose 'usb' or 'rpi'.")
    if not src_element:
        print(f"Error: Could not create source element for {camera_type}.")
        return None
    return src_element

def run_gst_pipeline():
    """Initializes and runs the GStreamer pipeline in a separate thread."""
    global gst_pipeline, main_loop
    Gst.init(None)

    # --- Create GStreamer Elements ---
    source = get_source_element(CAMERA_TYPE)
    if not source:
        return # Error already printed

    capsfilter_src = Gst.ElementFactory.make("capsfilter", "capsfilter_src")
    videoconvert = Gst.ElementFactory.make("videoconvert", "videoconvert")
    videoscale = Gst.ElementFactory.make("videoscale", "videoscale")
    capsfilter_scale = Gst.ElementFactory.make("capsfilter", "capsfilter_scale")
    hailovideoscale = Gst.ElementFactory.make("hailovideoscale", "hailovideoscale")
    hailonet = Gst.ElementFactory.make("hailonet", "hailonet")
    hailofilter = Gst.ElementFactory.make("hailofilter", "hailofilter")
    appsink = Gst.ElementFactory.make("appsink", "appsink")

    if not all([capsfilter_src, videoconvert, videoscale, capsfilter_scale,
                hailovideoscale, hailonet, hailofilter, appsink]):
        print("Error: Not all GStreamer elements could be created.")
        return

    # --- Configure Elements ---
    # Source caps - adjust resolution/framerate if needed
    # Example: "video/x-raw,width=640,height=480,framerate=30/1"
    src_caps_str = "video/x-raw,width=640,height=480"
    src_caps = Gst.Caps.from_string(src_caps_str)
    capsfilter_src.set_property("caps", src_caps)

    # Scaling caps - match model input if necessary, otherwise just format
    scale_caps_str = "video/x-raw,format=RGB" # HailoNet usually expects RGB
    scale_caps = Gst.Caps.from_string(scale_caps_str)
    capsfilter_scale.set_property("caps", scale_caps)

    # HailoNet configuration
    if not os.path.exists(HEF_FILE_PATH):
        print(f"Error: HEF file not found at {HEF_FILE_PATH}")
        return
    hailonet.set_property("hef-path", HEF_FILE_PATH)

    # HailoFilter configuration (optional, depends on model postprocessing)
    # Example: hailofilter.set_property("config-path", "path/to/filter/config.json")

    # Appsink configuration
    appsink.set_property("emit-signals", True)
    appsink.set_property("sync", False) # Don't block pipeline if app is slow
    appsink.set_property("max-buffers", 1) # Process latest frame
    appsink.set_property("drop", True)
    appsink.connect("new-sample", app_callback) # Connect the callback

    # --- Create and Link Pipeline ---
    gst_pipeline = Gst.Pipeline.new("hailo-detection-pipeline")

    for element in [source, capsfilter_src, videoconvert, videoscale,
                    capsfilter_scale, hailovideoscale, hailonet,
                    hailofilter, appsink]:
        gst_pipeline.add(element)

    if not Gst.Element.link_many(source, capsfilter_src, videoconvert, videoscale,
                                 capsfilter_scale, hailovideoscale, hailonet,
                                 hailofilter, appsink):
        print("Error: Could not link GStreamer elements.")
        return

    # --- Start Pipeline ---
    gst_pipeline.set_state(Gst.State.PLAYING)
    print("GStreamer pipeline started.")

    # --- Run GLib Main Loop ---
    main_loop = GLib.MainLoop()
    try:
        main_loop.run()
    except KeyboardInterrupt:
        pass # Handled in main shutdown

    # --- Cleanup ---
    print("Stopping GStreamer pipeline...")
    if gst_pipeline:
        gst_pipeline.set_state(Gst.State.NULL)
    print("GStreamer pipeline stopped.")


def app_callback(appsink):
    """Callback function called by appsink when a new sample is available."""
    global latest_frame, latest_detections, data_lock
    sample = appsink.emit("pull-sample")
    if sample is None:
        return Gst.FlowReturn.OK

    buffer = sample.get_buffer()
    if buffer is None:
        return Gst.FlowReturn.OK

    # Get frame data
    caps = sample.get_caps()
    format, width, height = get_caps_from_pad(caps) # Use helper
    frame_data = None
    if format is not None and width is not None and height is not None:
        frame_data = get_numpy_from_buffer(buffer, format, width, height)

    # Get Hailo detection data
    current_detections = []
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    for detection in detections:
        label = detection.get_label() # Get label from the detection
        bbox = detection.get_bbox()   # Get BBox object
        confidence = detection.get_confidence()
        # Get coordinates - Hailo BBox provides normalized (0.0-1.0) values
        xmin_norm, ymin_norm = bbox.xmin(), bbox.ymin()
        width_norm, height_norm = bbox.width(), bbox.height()

        # Convert normalized coordinates to pixel coordinates if frame exists
        if frame_data is not None:
            h, w, _ = frame_data.shape
            x1 = int(xmin_norm * w)
            y1 = int(ymin_norm * h)
            x2 = int((xmin_norm + width_norm) * w)
            y2 = int((ymin_norm + height_norm) * h)

            # Draw bounding box and label on the frame (optional)
            cv2.rectangle(frame_data, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label_text = f"{label}: {confidence:.2f}"
            cv2.putText(frame_data, label_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Store detection info for websocket
            current_detections.append({
                "label": label,
                "confidence": float(confidence), # Ensure serializable
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "center_x": int((x1 + x2) / 2),
                "center_y": int((y1 + y2) / 2)
            })
        else:
             # Store normalized coordinates if frame data isn't available/processed
             current_detections.append({
                "label": label,
                "confidence": float(confidence),
                "xmin_norm": float(xmin_norm),
                "ymin_norm": float(ymin_norm),
                "width_norm": float(width_norm),
                "height_norm": float(height_norm)
            })


    # Update shared variables safely
    with data_lock:
        if frame_data is not None:
            # Convert RGB (from GStreamer) back to BGR for cv2 encoding
            latest_frame = cv2.cvtColor(frame_data, cv2.COLOR_RGB2BGR)
        else:
            latest_frame = None # Or handle case where frame isn't needed/available
        latest_detections = current_detections

    return Gst.FlowReturn.OK

# --- WebSocket Server Functions ---

def get_local_ip():
    """Gets the local IP address of the machine."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception:
        return "127.0.0.1"

async def process_frame_for_websocket(frame):
    """Encodes a CV2 frame to base64 JPEG."""
    if frame is None:
        return None
    ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80]) # Quality 80
    if not ret:
        return None
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
    return jpg_as_text

async def handle_client(websocket):
    """Handles an incoming websocket connection."""
    global latest_frame, latest_detections, data_lock
    print(f"Client connected from {websocket.remote_address}")
    try:
        while True:
            # Get the latest data safely
            with data_lock:
                frame_to_send = latest_frame.copy() if latest_frame is not None else None
                detections_to_send = list(latest_detections) # Shallow copy

            # Encode frame if available
            encoded_frame = await process_frame_for_websocket(frame_to_send)

            # Prepare message
            message = {
                "image": encoded_frame, # Will be None if frame couldn't be processed
                "detections": detections_to_send
            }

            # Send data
            await websocket.send(json.dumps(message))

            # Control send rate (adjust as needed)
            await asyncio.sleep(0.05) # ~20 FPS target send rate

    except websockets.exceptions.ConnectionClosed:
        print(f"Client disconnected: {websocket.remote_address}")
    except Exception as e:
        print(f"Error during websocket handling: {e}")
    finally:
        print("Client connection closed.")


async def main():
    """Starts the GStreamer pipeline thread and the WebSocket server."""
    global main_loop

    # Start GStreamer pipeline in a separate thread
    gst_thread = threading.Thread(target=run_gst_pipeline, daemon=True)
    gst_thread.start()

    # Wait a moment for GStreamer to potentially initialize
    await asyncio.sleep(2)

    # Check if pipeline started successfully (basic check)
    if not gst_pipeline or not main_loop:
         print("GStreamer pipeline failed to initialize. Exiting.")
         # Optionally signal the gst_thread to stop if it's stuck
         return

    # Get local IP and start WebSocket server
    local_ip = get_local_ip()
    port = 8765
    server = await websockets.serve(handle_client, "0.0.0.0", port)

    print(f"WebSocket server running at:")
    print(f"  • Local:   ws://localhost:{port}")
    print(f"  • Network: ws://{local_ip}:{port}")
    print("\nPress Ctrl+C to stop.")

    try:
        await server.wait_closed()
    except asyncio.CancelledError:
        print("WebSocket server stopping.")
    finally:
        # Signal GStreamer thread to stop
        if main_loop and main_loop.is_running():
            main_loop.quit()
        # Wait for GStreamer thread to finish cleanup
        gst_thread.join(timeout=5)
        print("Server shut down.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Application interrupted. Shutting down...")
