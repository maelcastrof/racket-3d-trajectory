# IACV Project racket-3d-trajectory

This project involves detecting and tracking a ping pong racket in videos using computer vision techniques. It includes camera calibration, color detection, and racket tracking.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/maelcastrof/racket-3d-trajectory.git
   cd racket-3d-trajectory.git
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
#### 1. Camera Calibration
- **Purpose**: Generate the calibration matrix and distortion coefficients (already done)
- **How to Run**:
  ```bash
  python pre-processing/calibration.py
  ```
- **Output**: Calibration results are saved in `pre-processing/calibration_results.py`.

#### 2. Identifying Racket Color
- **Purpose**: Identify the racket's HSV color range.
- **How to Run**:
  ```bash
  python pre-processing/racket_color.py
  ```
- **Output**: The HSV color range (e.g., `[173 149 106]`) is printed and saved in `pre-processing/calibration_results.py`.

#### 3. Racket Tracking
- **Purpose**: Detect and track the racket in the video based on its color.
- **How to Run**:
  ```bash
  python tracking_racket_color.py
  ```
- **Input**: Video file path is hardcoded in the script. Modify `video_path` in `tracking_racket_color.py` if needed.
- **Output**: Real-time tracking of the racket is displayed.