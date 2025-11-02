import cv2
import os
import numpy as np
import argparse

def find_crack_features(image_path, threshold_val=100, max_corners=500, quality=0.01, min_dist=5):
    """
    Detects key feature points along a crack using Shi-Tomasi.
    
    Args:
        image_path (str): Path to the source crack image.
        threshold_val (int): Pixel brightness value (0-255) to separate 
                             the crack from the background.
        max_corners (int): The maximum number of corners (points) to detect.
        quality (float): The quality level for corner detection (0-1).
        min_dist (int): The minimum Euclidean distance between corners.
    """
    
    # 1. Load Image
    print(f"Loading image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return
        
    # Create a copy to draw on for the final output
    output_image = image.copy()
    
    # 2. Pre-processing
    print("Pre-processing image...")
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply a Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Use an inverted binary threshold.
    # The crack is dark (low pixel value), so we threshold *below* threshold_val.
    # THRESH_BINARY_INV makes pixels below 'threshold_val' (the crack) white (255),
    # and all others (background) black (0).
    # Shi-Tomasi will find features on the white pixels.
    _value, thresh = cv2.threshold(blur, threshold_val, 255, cv2.THRESH_BINARY_INV)

    # 3. Detect Corners (Good Features to Track)
    print(f"Detecting {max_corners} features on the crack...")
    corners = cv2.goodFeaturesToTrack(
        thresh,
        maxCorners=max_corners,
        qualityLevel=quality,
        minDistance=min_dist
    )
    
    # 4. Visualize
    if corners is not None:
        print(f"Found {len(corners)} feature points.")
        # Reshape corners from [[[x, y]]] to (N, 2)
        corners = np.int8(corners).reshape(-1, 2)
        
        for (x, y) in corners:
            # Draw a small, bright red circle at each corner
            cv2.circle(output_image, (x, y), 3, (0, 0, 255), -1)
    else:
        print("No corners found. Try adjusting parameters (especially --thresh).")

    # 5. Save the final image
    # Extract original filename (e.g., "s62.jpg")
    base_name = image_path.split('/')[-1].split('\\')[-1]
    output_filename = f"detected_{base_name}"

    #Save the output image inside a new folder called "output_images"
    os.makedirs("output_images", exist_ok=True)
    output_path = os.path.join("output_images", output_filename)
    cv2.imwrite(output_path, output_image)
    print(f"Output image saved to {output_path}")
    
    # Optional: Display the results
    cv2.imshow('Original Image', image)
    cv2.imshow('Thresholded Crack (Features are found here)', thresh)
    cv2.imshow('Crack Features Detected', output_image)
    
    print("Press any key to close the windows...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# --- Main execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect features on crack images using Shi-Tomasi.")
    parser.add_argument("image_path", type=str, help="Path to the input crack image.")
    parser.add_argument("--thresh", type=int, default=100, 
                        help="Threshold value (0-255). Lower values = more sensitive to dark areas.")
    parser.add_argument("--corners", type=int, default=500, 
                        help="Max number of corners to find.")
    parser.add_argument("--quality", type=float, default=0.01, 
                        help="Quality level for corners (0.01-1.0).")
    parser.add_argument("--min_dist", type=int, default=5, 
                        help="Minimum distance between corners.")
    
    args = parser.parse_args()
    
    find_crack_features(
        args.image_path, 
        args.thresh, 
        args.corners, 
        args.quality, 
        args.min_dist
    )