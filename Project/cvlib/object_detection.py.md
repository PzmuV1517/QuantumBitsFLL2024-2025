# Internal Code Documentation: Object Detection and Drowning Alert System

## Table of Contents

1. [Introduction](#introduction)
2. [Module Imports and Global Variables](#module-imports-and-global-variables)
3. [Function `populate_class_labels()`](#function-populate_class_labels)
4. [Function `get_output_layers()`](#function-get_output_layers)
5. [Function `draw_bbox()`](#function-draw_bbox)
6. [Function `detect_common_objects()`](#function-detect_common-objects)
    * [YOLO Object Detection Algorithm](#yolo-object-detection-algorithm)
7. [Function `play_sound()`](#function-play_sound)


<a name="introduction"></a>
## 1. Introduction

This document provides internal code documentation for a Python-based object detection system with a drowning alert functionality. The system utilizes the YOLOv3 object detection model to identify persons in an image and triggers an alarm if a person is classified as potentially drowning.


<a name="module-imports-and-global-variables"></a>
## 2. Module Imports and Global Variables

The code begins by importing necessary packages:

| Package        | Purpose                                         |
|----------------|-----------------------------------------------------|
| `cv2`          | OpenCV for image processing                      |
| `os`           | For operating system related functionalities       |
| `numpy`        | For numerical operations                           |
| `utils.download_file` | Custom utility function to download files          |
| `playsound`    | To play sound files                               |
| `threading`    | For managing threads                              |


Global variables are initialized:

| Variable       | Description                                               |
|----------------|-----------------------------------------------------------|
| `initialize`   | Boolean flag indicating if the YOLO model is initialized  |
| `net`          | OpenCV DNN object representing the YOLOv3 neural network |
| `dest_dir`     | Directory to store downloaded YOLO files                 |
| `classes`      | List of object class labels                             |
| `COLORS`       | List of colors (BGR) for bounding boxes                  |


<a name="function-populate_class-labels"></a>
## 3. Function `populate_class_labels()`

This function downloads and loads class labels from a text file if it doesn't exist locally.


```python
def populate_class_labels():
    # ... (function code) ...
```

The function checks if the class label file (`yolov3_classes.txt`) exists in the designated directory. If not, it downloads the file from a specified GitHub URL using the `download_file` utility function.  Then it reads each line from the file, strips whitespace, and returns a list of class labels.


<a name="function-get-output-layers"></a>
## 4. Function `get_output_layers()`

This function retrieves the output layer names from the YOLOv3 network.

```python
def get_output_layers(net):
    # ... (function code) ...
```

It takes the YOLOv3 network (`net`) as input and uses the `getLayerNames()` and `getUnconnectedOutLayers()` methods from OpenCV's DNN module to identify and return the names of the output layers.


<a name="function-draw-bbox"></a>
## 5. Function `draw_bbox()`

This function draws bounding boxes around detected objects on an image.

```python
def draw_bbox(img, bbox, labels, confidence, Drowning, write_conf=False):
    # ... (function code) ...
```

The function iterates through the detected objects. If an object is classified as a 'person' and the `Drowning` flag is True, it draws a red bounding box with "ALERT DROWNING" label and starts a new thread to play an alarm sound. Otherwise, it draws a green bounding box with "Normal" label.  Confidence scores can optionally be displayed.


<a name="function-detect-common-objects"></a>
## 6. Function `detect_common_objects()`

This function performs object detection using the YOLOv3 model.

```python
def detect_common_objects(image, confidence=0.5, nms_thresh=0.3):
    # ... (function code) ...
```

This function is the core of the object detection process.


### YOLO Object Detection Algorithm

1. **Download YOLO Files:** The function first checks if the YOLO configuration file (`yolov3.cfg`) and weights file (`yolov3.weights`) exist locally. If not, it downloads them from specified URLs.

2. **Initialize YOLO Model:** If the global variable `initialize` is True (meaning the model hasn't been loaded yet), the function loads the YOLOv3 model using `cv2.dnn.readNet()`, populates the class labels, and sets `initialize` to False.

3. **Create Blob:** The input image is preprocessed into a blob using `cv2.dnn.blobFromImage()`, which resizes the image to 416x416 pixels and normalizes pixel values.

4. **Forward Pass:** The blob is fed into the YOLOv3 network (`net.setInput(blob)`), and a forward pass is performed using `net.forward(get_output_layers(net))` to get detection results.

5. **Process Detections:** The function iterates through the detection results. For each detection with a confidence score above the specified threshold (`confidence`), the bounding box coordinates, class ID, and confidence score are extracted.

6. **Non-Maximum Suppression (NMS):**  `cv2.dnn.NMSBoxes()` performs non-maximum suppression to filter out overlapping bounding boxes, keeping only the ones with the highest confidence scores.

7. **Return Results:** Finally, the function returns a list of bounding boxes (`bbox`), corresponding labels (`label`), and confidence scores (`conf`).


<a name="function-play-sound"></a>
## 7. Function `play_sound()`

This function plays an alarm sound using the `playsound` library.

```python
def play_sound():
    playsound('sound/alarm.mp3')
```

It takes no arguments and plays the sound file `alarm.mp3` located in the `sound` directory.  Note that the file path should be adjusted if the sound file is located elsewhere.
