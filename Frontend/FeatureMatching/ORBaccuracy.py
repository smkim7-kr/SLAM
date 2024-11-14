"""
Accuracy-optimized ORB Feature Matching

Properties:
- More features (3000) for better matching chances
- Smaller scale factor (1.1) for more precise scale handling
- More pyramid levels (12) for better scale invariance
- Brute Force matcher with cross-checking for accuracy
- Two-stage matching with adaptive threshold
- RANSAC-based geometric verification
- Larger patch size for better feature description
- More sophisticated filtering using distance thresholds

Trade-offs:
+ More accurate and reliable matches
+ Better handling of viewpoint and scale changes
+ More robust against false positives
- Slower processing time
- Higher memory usage
- May be overkill for simple matching scenarios
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

def match_orb_features_accuracy(img1_path, img2_path):
    # Read and convert images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Accurate ORB settings
    orb = cv2.ORB_create(
        nfeatures=3000,        # More features for better matching
        scaleFactor=1.1,       # Smaller scale factor for more precise scale handling
        nlevels=12,            # More pyramid levels
        edgeThreshold=31,      # Larger edge threshold
        firstLevel=0,
        WTA_K=3,              # More points for BRIEF descriptor
        patchSize=41,         # Larger patch size for better description
        fastThreshold=15      # Lower threshold to detect more features
    )

    # Detect and compute
    start_time = time.time()
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    # Brute Force matcher with cross-checking
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    # Two-stage matching for better accuracy
    # First stage: basic matching
    matches = bf.match(des1, des2)
    
    # Second stage: distance-based filtering
    min_dist = min(m.distance for m in matches)
    max_dist = max(m.distance for m in matches)
    
    
    # Simple distance-based filtering
    matches = sorted(matches, key=lambda x: x.distance)
    good_matches = matches[:30]  # Just take top N matches
    
    # # Keep only the very best matches
    # good_matches = []
    # threshold = min_dist + 0.2 * (max_dist - min_dist)
    # for m in matches:
    #     if m.distance < threshold:
    #         good_matches.append(m)
            
    # Optional: Geometric verification using RANSAC
    if len(good_matches) > 10:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matches_mask = mask.ravel().tolist()
        
        # Keep only inlier matches
        good_matches = [m for i, m in enumerate(good_matches) if matches_mask[i]]
    
    end_time = time.time()

    # Visualization
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None,
                                 flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    img_matches = cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(16, 8))
    plt.imshow(img_matches)
    plt.title(f'Accuracy-optimized ORB matches (Time: {end_time-start_time:.3f}s)')
    plt.axis('off')
    plt.show()

    print(f"Processing time: {end_time-start_time:.3f} seconds")
    print(f"Keypoints in image 1: {len(kp1)}")
    print(f"Keypoints in image 2: {len(kp2)}")
    print(f"Initial matches: {len(matches)}")
    print(f"Good matches after filtering: {len(good_matches)}")


if __name__ == "__main__":
    img1_path = "data/cafe1.jpg"
    img2_path = "data/cafe2.jpg"
    match_orb_features_accuracy(img1_path, img2_path)