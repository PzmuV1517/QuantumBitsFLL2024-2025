# Real-time Drowning Detection System: Internal Code Documentation

## Table of Contents

1. [Introduction](#introduction)
2. [System Overview](#system-overview)
3. [Model Loading and Initialization](#model-loading-and-initialization)
4. [CustomCNN Class](#customcnn-class)
5. [Data Augmentation](#data-augmentation)
6. [Drowning Detection Algorithm (`detectDrowning` Function)](#drowning-detection-algorithm-detectdrowning-function)
    * [Single Person Detection](#single-person-detection)
    * [Multiple Person Detection](#multiple-person-detection)
7. [Bounding Box and Label Drawing](#bounding-box-and-label-drawing)


<a name="introduction"></a>
## 1. Introduction

This document details the implementation of a real-time drowning detection system using a custom Convolutional Neural Network (CNN) and object detection. The system analyzes video frames from a camera feed to identify potential drowning incidents.


<a name="system-overview"></a>
## 2. System Overview

The system consists of three primary components:

1. **Object Detection:**  Uses `cvlib` library to detect persons in each frame.
2. **Model-based Drowning Classification:** A custom CNN (`CustomCNN`) classifies a single detected person as either "drowning" or "not drowning".
3. **Logic-based Drowning Detection:** If multiple people are detected, proximity analysis is used to detect potential drowning situations.

The system processes frames from a camera feed in real-time, displaying results (bounding boxes, labels, and drowning status) overlaid on the video feed.


<a name="model-loading-and-initialization"></a>
## 3. Model Loading and Initialization

The system begins by loading a pre-trained label binarizer (`lb.pkl`) and a pre-trained CNN model (`model.pth`). The label binarizer maps predicted numerical classes to corresponding labels (e.g., "drowning", "not drowning"). The CNN model is loaded into a `CustomCNN` instance and set to evaluation mode.

```python
# Load label binarizer and model
print('Loading model and label binarizer...')
lb = joblib.load('lb.pkl')

print('Model Loaded...')
model = CustomCNN()
model.load_state_dict(torch.load('model.pth', map_location='cpu'))
model.eval()
print('Loaded model state_dict...')
```


<a name="customcnn-class"></a>
## 4. CustomCNN Class

This class defines the architecture of the custom CNN used for drowning classification.

| Layer        | Description                                          | Input Channels | Output Channels | Kernel Size | Pooling |
|--------------|------------------------------------------------------|-----------------|-----------------|-------------|---------|
| `conv1`      | 2D Convolutional layer                               | 3               | 16              | 5           | 2x2     |
| `conv2`      | 2D Convolutional layer                               | 16              | 32              | 5           | 2x2     |
| `conv3`      | 2D Convolutional layer                               | 32              | 64              | 3           | 2x2     |
| `conv4`      | 2D Convolutional layer                               | 64              | 128             | 5           | 2x2     |
| `fc1`        | Fully connected layer                                 | 128             | 256             | -           | -       |
| `fc2`        | Fully connected layer (output layer)                   | 256             | Number of Classes | -           | -       |

The `forward` method defines the forward pass through the network.  It applies convolutional layers with ReLU activation, max pooling, and finally, adaptive average pooling to reduce the spatial dimensions before passing the data through fully connected layers. The output is a vector of probabilities for each class.


```python
class CustomCNN(nn.Module):
    def __init__(self):
        # ... (Layer definitions as shown in the table above) ...
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        bs, _, _, _ = x.shape
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```


<a name="data-augmentation"></a>
## 5. Data Augmentation

Albumentations library is used to resize images to 224x224 pixels before feeding them to the CNN model.


```python
aug = albumentations.Compose([
    albumentations.Resize(224, 224),
])
```


<a name="drowning-detection-algorithm-detectdrowning-function"></a>
## 6. Drowning Detection Algorithm (`detectDrowning` Function)

The core drowning detection logic resides within the `detectDrowning` function.  It handles both single and multiple person detection scenarios.

<a name="single-person-detection"></a>
### 6.1 Single Person Detection

If only one person is detected, the system uses the pre-trained CNN model for classification:

1. The bounding box coordinates of the person are extracted.
2. The region of interest (ROI) is cropped from the frame.
3. The ROI is preprocessed (resized, converted to tensor, and normalized).
4. The preprocessed image is passed through the CNN model to obtain a prediction.
5. The prediction is mapped to a label ("drowning" or "not drowning") using the label binarizer.

<a name="multiple-person-detection"></a>
### 6.2 Multiple Person Detection

If multiple persons are detected, a simple proximity-based approach is used. The system calculates the distances between the centers of all bounding boxes. If the minimum distance between any two persons is less than 50 pixels, it's flagged as a potential drowning situation (assuming people are close together during a rescue attempt).


<a name="bounding-box-and-label-drawing"></a>
## 7. Bounding Box and Label Drawing

The `draw_bbox` function from `cvlib` is used to draw bounding boxes around detected persons and display labels indicating their classification ("drowning" or "not drowning"). The `isDrowning` boolean variable determines the color of the bounding box (likely red for drowning and green otherwise).  This function is called after the drowning detection logic is executed, regardless of whether single or multiple person detection was used.
