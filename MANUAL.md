# AiOnDrone Project Manual

## Table of Contents

- [Project Overview](#project-overview)
- [Technical Details](#technical-details)
  - [Core Components](#core-components)
  - [AI Implementation](#ai-implementation)
  - [Drone Control](#drone-control)
  - [Object Detection](#object-detection)
  - [Drowning Detection](#drowning-detection)
  - [Group Photo Functionality](#group-photo-functionality)
  - [Path Planning](#path-planning)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Basic Usage](#basic-usage)
  - [Advanced Features](#advanced-features)
- [Troubleshooting](#troubleshooting)
- [Future Development](#future-development)
- [References](#references)

## Project Overview

AiOnDrone is a comprehensive drone control and automation project that integrates artificial intelligence capabilities with unmanned aerial vehicles (UAVs). The project aims to enhance drone functionality through advanced computer vision techniques, autonomous navigation, and specialized features for safety and utility applications. It's particularly focused on water safety, with specialized drowning detection capabilities, while also providing general-purpose object detection and drone control functions.

The system comprises several integrated modules that work together to provide drone control, video processing, object detection, path planning, and specialized applications like drowning detection and group photo capturing. The project leverages modern AI techniques, particularly YOLO (You Only Look Once) models, to enable real-time object detection and analysis of video streams.

Throughout its history, the project has maintained a focus on creating a modular, extensible platform that can be adapted for various applications while maintaining core functionality for drone control and AI-based video analysis.

## Technical Details

### Core Components

The AiOnDrone project consists of several key components:

1. **Main Controller (`AiOnDrone.py`)**: The central module that coordinates the interaction between the drone hardware, AI systems, and specialized modules.

2. **AI Testing Module (`aitest.py`)**: Provides functionality to test AI features separate from drone hardware, enabling development and debugging of AI capabilities.

3. **Drone Control System**: Multiple modules (`manualcontroll.py`, `dronecontroll.py`, `dronetest.py`) that handle different aspects of drone operation, from manual user control to programmatic commands.

4. **Computer Vision Library (`cvlib/`)**: A collection of modules for image processing and object detection using computer vision techniques.

5. **Specialized Applications**: Purpose-built modules for specific tasks such as drowning detection (`DrownDetect.py`), group photos (`groupphoto.py`), and path planning (`path-plan.py`).

6. **Battery Management (`battery.py`)**: Monitors and manages drone battery levels to ensure safe operation and provide warnings when power is low.

7. **AI Models**: Pre-trained models for different detection tasks (`drowning.pt`, `logo.pt`, `logoColor.pt`, `yolo11n.pt`), which use the YOLO (You Only Look Once) architecture for efficient object detection.

### AI Implementation

The AI system in AiOnDrone is built around YOLO (You Only Look Once) object detection models. These models provide real-time detection capabilities with high accuracy, making them suitable for drone applications where processing power may be limited, and real-time performance is critical.

Key AI components include:

1. **YOLO Models**: The project uses several custom-trained YOLO models:
   - `drowning.pt`: Specialized for detecting drowning persons in water
   - `yolo11n.pt`: General object detection model for identifying common objects
   - `logo.pt` and `logoColor.pt`: Models for logo detection, potentially for navigation or landing zone identification

2. **Object Detection Pipeline**: The detection process is implemented in the `cvlib/object_detection.py` module, which provides the `detect_common_objects` function. This function processes video frames to identify objects of interest.

3. **Model Training**: The models were trained on custom datasets (stored in `trainingdata/`) tailored to the specific application domains, particularly for drowning detection which requires specialized training data.

4. **Inference Optimization**: The system includes optimizations to ensure real-time performance on the limited computing resources available on drone platforms.

### Drone Control

The drone control system provides multiple layers of functionality:

1. **Manual Control (`manualcontroll.py`)**: Implements a user interface for direct control of the drone, allowing for manual flight control through keyboard or joystick input.

2. **Programmatic Control (`dronecontroll.py`)**: Provides an API for programmatic control of the drone, used by autonomous features and specialized applications.

3. **Testing and Calibration (`dronetest.py`, `dronetestcolor.py`)**: Modules for testing drone functionality and calibrating sensors and controls.

4. **Central Control (`dronecenter.py`)**: Coordinates between different control methods and manages priority of commands when multiple systems are active.

5. **Special Maneuvers (`droneflip.py`)**: Implements specialized flight maneuvers, such as flips or rotations, for specific use cases.

The control system is designed to be modular, allowing different control methods to be used interchangeably depending on the application requirements.

### Object Detection

Object detection is a core capability of the AiOnDrone system, implemented using the YOLO architecture:

1. **Detection Architecture**: The system uses a YOLO variant that balances accuracy and computational efficiency, suitable for the limited resources available on drone platforms.

2. **Model Architecture**: The YOLO models are structured according to the specifications in `cvlib/data/deploy.prototxt`, which defines the neural network architecture with layers such as:
   - Convolutional layers for feature extraction
   - Batch normalization and scaling for training stability
   - ReLU activation functions
   - Detection output layers for generating bounding boxes and classifications

3. **Detection Process**:
   - Input frames are captured from the drone's camera
   - Each frame is preprocessed and fed to the YOLO model
   - The model outputs detection results with bounding boxes, class IDs, and confidence scores
   - Post-processing is applied to filter and interpret the results

4. **Integration**: The detection system is integrated with other modules, allowing detections to trigger actions such as following detected objects, sending alerts, or capturing photos.

### Drowning Detection

The drowning detection system (`DrownDetect.py`) is a specialized application of the object detection capabilities:

1. **Custom Model**: A specialized YOLO model (`drowning.pt`) trained on datasets of drowning and swimming persons to differentiate between normal swimming behavior and drowning signs.

2. **Detection Algorithm**: The algorithm looks for specific body postures and movement patterns that indicate a person in distress in water.

3. **Alert System**: When a potential drowning is detected, the system can generate alerts through visual indicators, sound notifications, or communication to operators.

4. **Integration with Drone Control**: The drowning detection system can guide the drone to hover near the detected person or follow them to maintain visual contact while help arrives.

### Group Photo Functionality

The group photo module (`groupphoto.py`) provides specialized functionality for capturing photos of groups:

1. **Person Detection**: Uses the general object detection capabilities to identify and count people in the frame.

2. **Optimal Positioning**: Algorithms to determine the optimal distance and angle for capturing the entire group.

3. **Automated Capture**: Once positioned correctly, the system automatically captures photos with appropriate timing.

4. **Multiple Shot Options**: Capability to take multiple photos with slight variations in position to ensure the best possible group photo.

### Path Planning

The path planning system (`path-plan.py`) enables autonomous navigation:

1. **Waypoint Navigation**: Reads waypoint data from `waypoint.json` and calculates optimal paths between points.

2. **Obstacle Avoidance**: Integrates with the object detection system to identify and avoid obstacles in the flight path.

3. **Mission Planning**: Allows the definition of complete missions with multiple waypoints and actions to be performed at each point.

4. **Dynamic Replanning**: Capability to adjust paths in real-time based on detected obstacles or changing conditions.

## System Requirements

### Hardware Requirements

1. **Drone Platform**:
   - Compatible drone with programmable flight controller
   - Onboard camera with live video feed capability
   - Sufficient payload capacity for additional computing hardware (if required)

2. **Computing Hardware**:
   - Onboard computer (such as Raspberry Pi or NVIDIA Jetson) for AI processing
   - Minimum 2GB RAM (4GB or more recommended)
   - GPU or TPU acceleration recommended for optimal performance

3. **Sensors**:
   - Camera with minimum 720p resolution (1080p recommended)
   - Optional: Depth sensor for improved obstacle avoidance
   - Optional: GPS module for precise positioning

### Software Requirements

1. **Operating System**:
   - Linux-based system (Ubuntu 18.04 or later recommended)
   - Compatible with Python 3.6 or later

2. **Dependencies**:
   - Python 3.6+ with packages specified in `requirements.txt`
   - OpenCV for image processing
   - PyTorch for AI model inference
   - Additional libraries for drone communication and control

3. **Storage**:
   - Minimum 2GB for software and AI models
   - Additional storage for recorded video and logs

## Installation

The installation process involves setting up both hardware and software components:

1. **Clone the Repository**:

```git clone https://github.com/yourusername/AiOnDrone.git cd AiOnDrone```

2. **Set Up Virtual Environment**:

```python3 -m venv env source env/bin/activate # On Windows: env\Scripts\activate```

3. **Install Dependencies**:

```pip install -r requirements.txt```

4. **Download Pre-trained Models**:
- Ensure the following model files are in the Project directory:
  - `drowning.pt`
  - `logo.pt`
  - `logoColor.pt`
  - `yolo11n.pt`

5. **Configure Hardware**:
- Connect to the drone using the appropriate protocol
- Set up camera feed access
- Configure any additional sensors

6. **Test Installation**:

```python aitest.py```

This will verify that the AI components are working correctly without requiring drone hardware.

## Configuration

The AiOnDrone system is highly configurable through several configuration files and parameters:

1. **Waypoint Configuration** (`waypoint.json`):
- Define navigation waypoints for autonomous flight
- Specify actions to be performed at each waypoint
- Set altitude, speed, and other flight parameters

2. **AI Model Configuration**:
- Adjust detection thresholds in the respective modules
- Select which models to use for different scenarios
- Configure inference parameters for optimal performance

3. **Drone Control Configuration**:
- Set control sensitivity and response curves
- Configure safety limits for speed, altitude, and distance
- Adjust autonomous behavior parameters

4. **System Configuration**:
- Set video recording parameters (resolution, framerate, storage location)
- Configure alert and notification settings
- Set logging verbosity and storage options

## Usage

### Basic Usage

1. **Starting the System**:

```python AiOnDrone.py```

This launches the main application with default settings.

2. **Manual Control**:

```python manualcontroll.py```

Launches the manual control interface for direct flight control.

3. **Drowning Detection**:

```python DrownDetect.py```

Activates the specialized drowning detection mode.

4. **Group Photo**:

```python groupphoto.py```

Starts the group photo mode for automated group photography.

5. **Path Planning**:

```python path-plan.py```

Launches the autonomous navigation mode using predefined waypoints.

### Advanced Features

1. **Combined Modes**:
The system allows for combining multiple features, such as running drowning detection while following a predefined path.

2. **Custom Object Detection**:
Users can define custom objects to be detected by modifying the appropriate configuration files and using compatible YOLO models.

3. **Real-time Adjustments**:
During operation, various parameters can be adjusted in real-time through the control interface.

4. **Log Analysis**:
The system generates detailed logs (`drone_log.txt`) that can be analyzed for performance optimization and issue diagnosis.

## Troubleshooting

Common issues and their solutions:

1. **Connection Problems**:
- Ensure the drone is powered and within range
- Check that the correct communication protocol is configured
- Verify that no other software is attempting to control the drone

2. **Detection Issues**:
- Ensure lighting conditions are adequate for the camera
- Check that the appropriate model is being used for the detection task
- Adjust detection thresholds if false positives or negatives occur

3. **Performance Problems**:
- Lower video resolution if processing is too slow
- Ensure the computing hardware meets minimum requirements
- Close unnecessary applications to free up resources

4. **Battery Management**:
- Monitor battery levels closely and ensure automatic return-to-home is configured
- Keep spare batteries charged and ready
- Be aware that AI processing may increase power consumption

5. **Software Errors**:
- Check the log file (`drone_log.txt`) for detailed error information
- Ensure all dependencies are correctly installed
- Verify that model files are present and not corrupted

## Future Development

Planned future enhancements for the AiOnDrone project:

1. **Enhanced AI Capabilities**:
- Multi-object tracking with persistent identities
- Behavior analysis for more nuanced detection
- Integration with larger language models for more sophisticated decision-making

2. **Hardware Improvements**:
- Support for additional drone platforms
- Integration with specialized sensors (thermal cameras, LIDAR)
- Edge computing optimizations for improved performance

3. **New Applications**:
- Search and rescue mission planning
- Environmental monitoring capabilities
- Agricultural inspection and analysis

4. **User Interface Enhancements**:
- Mobile application for remote control and monitoring
- Web interface for mission planning and review
- Augmented reality integration for enhanced visualization

5. **Collaborative Features**:
- Swarm capabilities for multiple drones working together
- Shared detection and tracking between multiple systems
- Cloud integration for data sharing and analysis

## References

1. Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). You Only Look Once: Unified, Real-Time Object Detection. CVPR 2016.

2. Bradski, G. (2000). The OpenCV Library. Dr. Dobb's Journal of Software Tools.

3. Quigley, M., Gerkey, B. P., Conley, K., Faust, J., Foote, T., Leibs, J., ... & Ng, A. Y. (2009, May). ROS: an open-source Robot Operating System. In ICRA Workshop on Open Source Software (Vol. 3, No. 3.2).

4. Bin Mohd Taib, N. A., Bade, A., & Thasarathan, H. (2021). Detection of Human Drowning Using AI in Surveillance Systems: A Review. In Journal of Physics: Conference Series (Vol. 1755, No. 1, p. 012002). IOP Publishing.

5. LaValle, S. M. (1998). Rapidly-exploring random trees: A new tool for path planning. Technical Report. Computer Science Department, Iowa State University.

6. Additional documentation on drone control, computer vision, and AI integration can be found in the project's `/docs` folder. (*note* : `/docs` no longer exists and has been merged into `README.md`)