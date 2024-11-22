# Internal Code Documentation: Tello Drone Waypoint Navigation

[Linked Table of Contents](#linked-table-of-contents)

## Linked Table of Contents

* [1. Introduction](#1-introduction)
* [2. System Architecture](#2-system-architecture)
* [3. Code Overview](#3-code-overview)
    * [3.1 JSON Data Loading](#31-json-data-loading)
    * [3.2 Tello Drone Initialization](#32-tello-drone-initialization)
    * [3.3 Speed Control Mechanism](#33-speed-control-mechanism)
    * [3.4 Waypoint Execution Loop](#34-waypoint-execution-loop)
    * [3.5 Error Handling](#35-error-handling)
* [4. Algorithm Details](#4-algorithm-details)
    * [4.1 Waypoint Navigation Algorithm](#41-waypoint-navigation-algorithm)


## 1. Introduction

This document details the implementation of a waypoint navigation system for a DJI Tello drone. The system reads waypoint coordinates from a JSON file, controls the drone's movement to each waypoint, and incorporates speed control for smoother navigation.

## 2. System Architecture

The system consists of three main components:

1. **JSON Data Parser:** Reads waypoint data from a `waypoint.json` file.  The file is expected to contain a list of waypoints, each with a distance (`dist_cm`) and angle (`angle_deg`).

2. **Tello Drone Controller:** Uses the `djitellopy` library to interface with the Tello drone, sending commands for movement and rotation.

3. **Waypoint Navigation Algorithm:**  A control loop iterates through waypoints, managing the drone’s movement and speed based on the distance to the target and the defined thresholds.


## 3. Code Overview

### 3.1 JSON Data Loading

The system begins by loading waypoint data from a JSON file (`waypoint.json`). The file is expected to have a structure like this:

```json
{
  "wp": [
    {"dist_cm": 100, "angle_deg": 0},
    {"dist_cm": 50, "angle_deg": 90},
    {"dist_cm": 150, "angle_deg": -45}
  ]
}
```

The code uses `json.load()` to parse this data into a Python dictionary, extracting the waypoint list into the `waypoints` variable.


### 3.2 Tello Drone Initialization

The `djitellopy` library is used to connect to and control the Tello drone. The `tello.connect()` function establishes a connection, and the battery level is then displayed. The drone is commanded to take off using `tello.takeoff()`.


### 3.3 Speed Control Mechanism

The code implements a dynamic speed control system to ensure smooth transitions between waypoints.  Three variables govern this:

| Variable Name             | Description                                    | Value | Unit |
|--------------------------|------------------------------------------------|-------|------|
| `default_speed`           | Default speed of the drone.                     | 65    | cm/s |
| `near_speed`              | Reduced speed when nearing a waypoint.           | 20    | cm/s |
| `deceleration_threshold` | Distance at which the drone begins to decelerate.| 30    | cm   |

The drone initially moves at `default_speed`. As it approaches a waypoint (within `deceleration_threshold`), it slows down to `near_speed` to ensure a precise stop.


### 3.4 Waypoint Execution Loop

The core logic resides in the `for` loop iterating through the `waypoints` list.  Each waypoint is processed as follows:

1. **Distance Calculation:** The distance to travel (`distance`) and the rotation angle (`angle`) are extracted from the waypoint data.

2. **Incremental Movement:** The drone moves towards the waypoint in steps.  A `step_size` of 100 cm is used, but the actual move distance is adjusted to not overshoot the target. The `min()` function ensures that `move_distance` never exceeds `remaining_distance` or the `step_size`.

3. **Dynamic Speed Adjustment:**  The drone's speed is adjusted based on the `remaining_distance` and the `deceleration_threshold`.

4. **Rotation:** After reaching a waypoint, the drone rotates by the specified angle using `tello.rotate_clockwise()` or `tello.rotate_counter_clockwise()`, depending on the sign of the angle.

5. **Stabilization:** `time.sleep(1)` pauses execution briefly after each movement and rotation to allow the drone to stabilize.


### 3.5 Error Handling

A `try...except` block is used to catch any exceptions that may occur during waypoint execution.  The error is logged to the console, providing information for debugging.


## 4. Algorithm Details

### 4.1 Waypoint Navigation Algorithm

The waypoint navigation algorithm utilizes a simple incremental approach.  Instead of directly moving the drone the full distance to each waypoint, it breaks down the movement into smaller steps (`step_size`). This approach enhances stability and precision, particularly considering the potential for minor inaccuracies in the drone's movement.  The dynamic speed adjustment further refines the navigation by ensuring smooth deceleration as the drone approaches each waypoint, minimizing overshooting and improving accuracy at the target location.  The algorithm can be summarized as follows:

1. **Initialization:** Load waypoints from JSON, initialize drone.
2. **Iteration:** For each waypoint:
    * Calculate distance to travel.
    * While distance > 0:
        * Determine move distance (minimum of `step_size` and remaining distance).
        * Adjust speed based on proximity to waypoint.
        * Move drone forward by `move_distance`.
        * Update remaining distance.
        * Pause for stabilization.
    * Rotate drone if necessary.
    * Pause for stabilization.
3. **Landing:** Land the drone once all waypoints are processed.

This incremental approach provides robustness and adaptability, allowing for successful waypoint navigation even with minor variations in the drone's actual movement.
