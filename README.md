# ShiTomasi-Trace
Demonstration of Shi-Tomasi Corner Detection for payment crack images 

This Python script uses the **Shi-Tomasi Corner Detection** algorithm (`cv2.goodFeaturesToTrack`) to detect and visualize key feature points along cracks in concrete, bridges, or other surfaces.

This is a common first step in structural health monitoring (SHM) for tasks like crack path mapping, length estimation, and temporal analysis (tracking crack growth).

## How It Works

1.  **Load Image:** The script reads a source image.
2.  **Pre-processing:** To isolate the crack, it converts the image to grayscale, applies a Gaussian blur to reduce noise, and then uses an inverted binary threshold to create a black-and-white mask where the crack is represented by white pixels.
3.  **Feature Detection:** The Shi-Tomasi "Good Features to Track" algorithm is run on the thresholded image. It finds points of high "cornerness," which correspond to the bends, turns, and endpoints of the crack.
4.  **Visualization:** The script draws small circles on the *original* image at the location of each detected feature point, creating a clear map of the crack's structure.

## Setup

1.  Clone this repository:
    ```bash
    git clone https://github.com/Parikshith-S/ShiTomasi-Trace.git
    cd ShiTomasi-Trace
    ```

2.  Install the required Python libraries:
    ```bash
    pip install opencv-python numpy
    ```

## Usage

Run the script from your terminal, providing the path to your image.

```bash
python src/detect_crack_features.py data/s81.jpg --thresh 100 --corners 1000 --min_dist 3
```
or generally
```bash
python src/detect_crack_features.py <path_to_your_image> [options]
```

