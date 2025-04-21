import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstApp', '1.0')
from gi.repository import Gst, GstApp, GLib, GObject

import hailo # Hailo python library

import cv2
import asyncio
import websockets
import base64
import json
import numpy as np
import socket
import os # <-- Add import os

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
async def process_frame_async(frame):
    # Run encoding in executor to avoid blocking asyncio loop
    loop = asyncio.get_event_loop()
    _, buffer = await loop.run_in_executor(None, cv2.imencode, '.jpg', frame)
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
    return jpg_as_text

# GStreamer Callback
def on_new_sample(appsink, user_data):
    """Callback function for the appsink's 'new-sample' signal"""
    websocket = user_data['websocket']
    loop = user_data['loop']
    pipeline_config = user_data['pipeline_config']

    sample = appsink.pull_sample()
    if sample is None:
        print("Warning: Pulled None sample from appsink")
        return Gst.FlowReturn.OK # Continue pipeline processing

    buffer = sample.get_buffer()
    if buffer is None:
        print("Warning: Sample contained no buffer")
        return Gst.FlowReturn.OK

    caps = sample.get_caps()
    if caps is None:
        print("Warning: Sample contained no caps")
        return Gst.FlowReturn.OK

    # Extract frame properties from caps
    struct = caps.get_structure(0)
    width = struct.get_value("width")
    height = struct.get_value("height")

    # Map buffer to read data
    success, map_info = buffer.map(Gst.MapFlags.READ)
    if not success:
        print("Error: Failed to map buffer")
        # Consider returning FlowReturn.ERROR, but OK might prevent pipeline stall
        return Gst.FlowReturn.OK

    try:
        # Create numpy array from buffer data (assuming RGB format from pipeline)
        # Make a copy because the buffer will be unmapped
        frame = np.ndarray((height, width, 3), buffer=map_info.data, dtype=np.uint8).copy()

        # Process Hailo Detections
        logo_detected = False
        logo_boxes = []
        roi = hailo.get_roi_from_buffer(buffer) # Get Hailo ROI object
        detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

        for detection in detections:
            label = detection.get_label()
            # *** IMPORTANT: Replace 'person' with the actual label from your HEF model if not using standard yolov8s ***
            # Convert label to lower case for case-insensitive comparison
            # if label and label.lower() == pipeline_config['target_label']: # Original 'logo' check
            if label and label.lower() == 'person': # Example using 'person' for standard yolov8s.hef
                logo_detected = True # Keep variable name for compatibility, or rename if preferred
                bbox = detection.get_bbox() # Get bounding box (normalized coordinates)
                confidence = detection.get_confidence()

                # Convert normalized coordinates to pixel coordinates
                x1 = int(bbox.xmin() * width)
                y1 = int(bbox.ymin() * height)
                x2 = int(bbox.xmax() * width)
                y2 = int(bbox.ymax() * height)

                logo_boxes.append({
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "center_x": int((x1 + x2) / 2),
                    "center_y": int((y1 + y2) / 2),
                    "confidence": float(confidence) # Include confidence
                })

                # Optional: Draw bounding box and label on the frame for visualization
                # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    finally:
        # Ensure buffer is unmapped
        buffer.unmap(map_info)

    # Prepare and Send Data Asynchronously
    async def send_data():
        try:
            # Encode frame to Base64 asynchronously
            encoded_frame = await process_frame_async(frame)

            # Prepare message payload
            # Using original keys 'drowning_detected'/'drowning_boxes' for compatibility
            # Consider changing keys if client expects 'logo_detected'/'logo_boxes'
            message = {
                "image": encoded_frame,
                "drowning_detected": logo_detected, # Variable name kept for compatibility
                "drowning_boxes": logo_boxes      # Variable name kept for compatibility
            }

            # Send message over websocket
            if websocket.open:
                await websocket.send(json.dumps(message))
            else:
                print("Websocket closed, cannot send message.")

        except Exception as e:
            print(f"Error during data preparation or sending: {e}")

    # Schedule the async send_data function in the main asyncio event loop
    if loop.is_running():
        asyncio.run_coroutine_threadsafe(send_data(), loop)
    else:
        print("Event loop not running, cannot schedule send.")

    return Gst.FlowReturn.OK # Indicate success

# Handle websocket connections
async def handle_client(websocket):
    """Handle incoming websocket connections using GStreamer and Hailo"""
    print(f"Client connected from {websocket.remote_address}")
    pipeline = None
    loop = asyncio.get_event_loop() # Get the current asyncio event loop

    # --- Configuration ---
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # *** IMPORTANT: Update these paths if the files are located elsewhere ***
    # Construct paths relative to the script directory (Project/)
    # Assumes DroneCamera/ is one level up and contains resources/
    # If using your custom 'logov8s.hef', ensure it's placed correctly or update path.
    # Using standard 'yolov8s.hef' from DroneCamera/resources/ as an example:
    hef_file_path = os.path.join(script_dir, '..', 'DroneCamera', 'resources', 'yolov8s.hef')
    # hef_file_path = os.path.join(script_dir, 'logov8s.hef') # Original path for custom model

    # Let GStreamer find the .so file using environment variables (TAPPAS_POST_PROC_DIR)
    # set by setup_env.sh or standard Hailo installation paths.
    # No need to define so_file_path here if environment is set up correctly.
    # so_file_path = os.path.join(script_dir, 'libyolov8.so') # Original path assumption

    pipeline_config = {
        'device': '/dev/video0', # Webcam device
        'hef_path': hef_file_path, # Use corrected path
        # 'postprocess_so': so_file_path, # Removed explicit path
        'postprocess_so_name': 'libyolov8.so', # Provide only the name
        # 'target_label': 'logo', # Original label for custom model
        'target_label': 'person', # Example label for standard yolov8s. Adjust if using custom model.
        'use_hailooverlay': False # Set to True to let Hailo draw boxes
    }

    # Check if HEF file exists before creating pipeline
    if not os.path.exists(pipeline_config['hef_path']):
        print(f"Error: HEF file not found at {pipeline_config['hef_path']}")
        await websocket.close()
        return
    # SO file existence check removed - relying on GStreamer/Hailo environment path


    try:
        # Define GStreamer Pipeline String
        # Build the pipeline string step-by-step for clarity
        pipeline_elements = [
            f"v4l2src device={pipeline_config['device']}",
            "! videoconvert",
            # Ensure format is compatible with Hailo (often RGB or BGRx)
            "! video/x-raw,format=RGB",
            # Use queue for buffering between elements/threads
            "! queue",
            # Hailo inference element using the specified HEF
            f"! hailonet hef-path=\"{pipeline_config['hef_path']}\"", # Quote path
            "! queue",
            # Hailo post-processing filter using the specified .so library name
            # GStreamer should find it via TAPPAS_POST_PROC_DIR or standard paths
            f"! hailofilter so-path=\"{pipeline_config['postprocess_so_name']}\" qos=false",
        ]
        # Optional: Add hailooverlay for automatic drawing
        if pipeline_config['use_hailooverlay']:
             pipeline_elements.extend([
                "! queue",
                "! hailooverlay qos=false",
             ])
        # Add appsink to capture the output
        pipeline_elements.extend([
            "! queue",
            # Appsink configuration: emit signals, keep only latest buffer, drop old ones
            "! appsink name=sink emit-signals=true max-buffers=1 drop=true sync=false"
        ])

        pipeline_str = " ".join(pipeline_elements)
        print(f"Using GStreamer pipeline: {pipeline_str}")

        # Initialize GStreamer (safe to call multiple times)
        Gst.init(None)

        # Parse the pipeline string
        print("Creating GStreamer pipeline...")
        pipeline = Gst.parse_launch(pipeline_str)
        if not pipeline:
            print("Error: Failed to parse GStreamer pipeline.")
            await websocket.close()
            return

        # Get the appsink element by name
        appsink = pipeline.get_by_name('sink')
        if not appsink:
            print("Error: Could not find appsink element named 'sink'")
            pipeline.set_state(Gst.State.NULL) # Clean up pipeline
            await websocket.close()
            return

        # Set appsink properties and connect the callback
        appsink.set_property('emit-signals', True)
        # Pass websocket, loop, and config to the callback
        user_data = {'websocket': websocket, 'loop': loop, 'pipeline_config': pipeline_config}
        handler_id = appsink.connect('new-sample', on_new_sample, user_data)
        if handler_id == 0:
             print("Error: Failed to connect 'new-sample' signal to appsink.")
             pipeline.set_state(Gst.State.NULL)
             await websocket.close()
             return

        # Start the pipeline
        print("Starting GStreamer pipeline...")
        ret = pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            print("Error: Unable to set the pipeline to the playing state.")
            # Attempt cleanup even on failure
            pipeline.set_state(Gst.State.NULL)
            await websocket.close()
            return
        elif ret == Gst.StateChangeReturn.ASYNC:
            print("Pipeline state change is asynchronous.")
            # Optionally wait for state change completion, but often not needed for PLAYING

        print("Pipeline running...")

        # Keep the connection alive while the websocket is open
        # Data sending is handled by the GStreamer callback
        while websocket.open:
            # Check for GStreamer bus messages (e.g., errors, EOS)
            bus = pipeline.get_bus()
            msg = bus.timed_pop_filtered(10 * Gst.MSECOND, Gst.MessageType.ERROR | Gst.MessageType.EOS)
            if msg:
                if msg.type == Gst.MessageType.ERROR:
                    err, debug_info = msg.parse_error()
                    print(f"GStreamer Error: {err}, {debug_info}")
                    break # Exit loop on error
                elif msg.type == Gst.MessageType.EOS:
                    print("GStreamer End-Of-Stream received.")
                    break # Exit loop on EOS
            # Yield control to asyncio event loop
            await asyncio.sleep(0.1)

    except websockets.exceptions.ConnectionClosedOK:
        print(f"Client {websocket.remote_address} disconnected normally.")
    except websockets.exceptions.ConnectionClosedError as e:
        print(f"Client {websocket.remote_address} disconnected with error: {e}")
    except GLib.Error as e:
        print(f"GStreamer GLib Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred in handle_client: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        print("Initiating cleanup...")
        if pipeline is not None:
            # Disconnect signal handler first
            if 'handler_id' in locals() and handler_id > 0:
                appsink = pipeline.get_by_name('sink')
                if appsink:
                    print("Disconnecting appsink signal...")
                    appsink.disconnect(handler_id)

            # Stop the pipeline
            print("Stopping GStreamer pipeline (setting state to NULL)...")
            pipeline.set_state(Gst.State.NULL)
            # Check final state
            state, _, _ = pipeline.get_state(Gst.CLOCK_TIME_NONE)
            print(f"Pipeline final state: {state.value_nick}")
            pipeline = None # Release reference

        if websocket and not websocket.closed:
            await websocket.close()
            print("WebSocket connection closed.")
        print(f"Cleanup complete for {websocket.remote_address}.")

# Main function to start the server
async def main():
    local_ip = get_local_ip()
    port = 8765

    # Initialize GObject threads for GStreamer integration with Python threads/asyncio
    # Needs to be called once before using GStreamer in threads
    GObject.threads_init()
    # Initialize GStreamer library itself
    Gst.init(None)

    print("Starting WebSocket server...")
    server = await websockets.serve(handle_client, "0.0.0.0", port)

    print(f"WebSocket server running and listening on:")
    print(f"  • Local:   ws://localhost:{port}")
    print(f"  • Network: ws://{local_ip}:{port}")
    print("\nUsing Hailo accelerator with specified HEF.")
    print("Press Ctrl+C to stop the server.")

    # Keep the server running until manually stopped
    await server.wait_closed()
    print("WebSocket server stopped.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Server stopped by user (Ctrl+C).")
