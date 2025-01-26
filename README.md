# Racket 3D Motion Tracking

## Objective

The main objective of this project is to measure the 3D motion of a racket during the time intervals before, during and after hitting the ball. 

The system tracks the racket through a sequence of images, locates it in 3D space (estimating its pose and orientation) and generates the estimated 3D position with respect to the table. For the positioning, it is possible to perform it in real time. 

## Folder Structure

### `pre-processing/`
This folder contains functions used for:
- Detecting the camera calibration parameters.
- Detecting the racket's color in HSV space.

### `racket_position/`
Contains modules responsible for:
- Localizing the racket in the 3D space.
- Tracking the racket's position over time.
- Estimating its 3D trajectory.

### `racket_orientation/`
Contains modules for:
- Estimating and plotting the racket's orientation at any given frame.
- Visualizing the racket's position relative to the table in 3D space.

## How to Use

### 1. Visualize the Video with Estimated Position and 3D Racket Plot

To visualize the input video along with the following information:
- The racket's estimated position (displayed in the top-left corner of the frame).
- The racket's position relative to the table in 3D space (shown in real-time at the bottom).

Run the following command:

```bash
python -m racket_position.main
```
### 2. Visualize the frame with Estimated Position and Orientation 3D Racket Plot
```bash
python -m racket_orientation.main
```