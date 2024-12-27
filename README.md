# IACV Project racket-3d-trajectory

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
### Step 1: Capture Calibration Images
1. Print a chessboard pattern and place it in various orientations and distances.
2. Capture images of the chessboard using your camera.
3. Save these images in the directory `img/chessboard/`.

### Step 2: Run Calibration
1. Run the calibration script:
   ```bash
   python calibration.py
   ```