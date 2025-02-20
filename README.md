# AiOnDrone Project

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Technical Details](#technical-details)
- [Contributing](#contributing)
- [License](#license)

## Introduction
AiOnDrone is a project that integrates artificial intelligence with drone technology to perform various tasks such as object detection, path planning, and manual control. The project leverages computer vision and machine learning models to enhance the capabilities of drones.

## Features
- Object detection using YOLO models
- Path planning for autonomous navigation
- Manual control interface
- Battery management
- Drowning detection
- Group photo capture
- Video recording and processing

## Installation
To install and set up the project, follow these steps:

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/AiOnDrone.git
    cd AiOnDrone/Project
    ```

2. Create and activate a virtual environment:
    ```sh
    python -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`
    ```

3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage
To use the project, you can run various scripts depending on the functionality you need:

- To run the AI on the drone:
    ```sh
    python AiOnDrone.py
    ```

- To test AI functionalities:
    ```sh
    python aitest.py
    ```

- To control the drone manually:
    ```sh
    python manualcontroll.py
    ```

- To detect drowning:
    ```sh
    python DrownDetect.py
    ```

- To capture a group photo:
    ```sh
    python groupphoto.py
    ```

- To plan a path:
    ```sh
    python path-plan.py
    ```

## Technical Details

### Object Detection
The object detection functionality is implemented using the YOLO (You Only Look Once) model. The model is loaded from the `drowning.pt` and `yolo11n.pt` files. The detection logic is encapsulated in the `detect_common_objects` function.

### Path Planning
Path planning is handled by the `path-plan.py` script, which uses algorithms to determine the optimal path for the drone to follow. The script reads waypoints from the `waypoint.json` file.

### Manual Control
Manual control of the drone is facilitated by the `manualcontroll.py` script. This script provides an interface for the user to control the drone's movements manually.

### Drowning Detection
The drowning detection functionality is implemented in the `DrownDetect.py` script. It uses a pre-trained model (`drowning.pt`) to identify potential drowning incidents in real-time.

### Battery Management
Battery management is handled by the `battery.py` script, which monitors the drone's battery levels and provides alerts when the battery is low.

### Video Processing
Video recording and processing are managed by the `output.mp4` file. The project includes scripts for capturing and processing video footage from the drone's camera.



## Contributing
We welcome contributions to the AiOnDrone project. To contribute, follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add some feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Open a pull request.

Please ensure your code adheres to the project's coding standards and includes appropriate tests.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
