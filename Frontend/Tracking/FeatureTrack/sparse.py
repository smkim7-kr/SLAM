import cv2
import numpy as np
import glob
from pathlib import Path
from collections import deque

def detect_features(image, max_corners=1000):
    """Detect good features to track using Shi-Tomasi corner detector"""
    corners = cv2.goodFeaturesToTrack(
        image,
        maxCorners=max_corners,
        qualityLevel=0.01,
        minDistance=10,
        blockSize=3
    )
    return corners if corners is not None else np.array([])

def main():
    # Get images from KITTI dataset
    image_dir = Path("../../../../KITTI/camera/03/image_0")
    image_paths = sorted(glob.glob(str(image_dir / "*.png")))
    
    # Read first frame
    old_frame = cv2.imread(image_paths[0], cv2.IMREAD_GRAYSCALE)
    
    # Colors for visualization
    POINT_COLOR = (0, 0, 255)  # Red in BGR
    FLOW_COLOR = (0, 255, 0)   # Green in BGR
    
    # Parameters for lucas kanade optical flow
    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    )
    
    # Initialize feature queue with larger size
    feature_queue = deque(maxlen=500)
    
    # Detect initial features and add first 500 to queue
    initial_points = detect_features(old_frame, max_corners=500)
    if len(initial_points) > 0:
        for point in initial_points:
            feature_queue.append(point.reshape(-1, 2))
    
    # Create a mask image for drawing purposes
    mask = np.zeros_like(cv2.imread(image_paths[0]))
    
    for image_path in image_paths[1:]:
        frame = cv2.imread(image_path)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if len(feature_queue) > 0:
            old_points = np.float32(list(feature_queue)).reshape(-1, 1, 2)
            
            # Calculate optical flow
            new_points, status, error = cv2.calcOpticalFlowPyrLK(
                old_frame, 
                frame_gray, 
                old_points, 
                None, 
                **lk_params
            )
            
            # Clear the queue and add back only the good points
            feature_queue.clear()
            good_new = new_points[status == 1]
            good_old = old_points[status == 1]
            
            for point in good_new:
                feature_queue.append(point.reshape(-1, 2))
        
        # Add new features if needed
        if len(feature_queue) < 500:
            new_features = detect_features(frame_gray, max_corners=500)
            if len(new_features) > 0:
                points_to_add = 500 - len(feature_queue)
                for point in new_features[:points_to_add]:
                    feature_queue.append(point.reshape(-1, 2))
        
        # Visualization
        mask = np.zeros_like(frame)  # Reset mask for each frame
        
        if len(good_new) > 0:
            for new, old in zip(good_new, good_old):
                a, b = new.ravel()
                c, d = old.ravel()
                # Draw flow lines in green
                mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), 
                               FLOW_COLOR, 2)
                # Draw current points in red
                frame = cv2.circle(frame, (int(a), int(b)), 2, 
                                 POINT_COLOR, -1)
        
        # Display the result
        img = cv2.add(frame, mask)
        cv2.imshow('Sparse Optical Flow', img)
        
        # Break on ESC key
        if cv2.waitKey(30) & 0xFF == 27:
            break
        
        # Update the previous frame
        old_frame = frame_gray.copy()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
