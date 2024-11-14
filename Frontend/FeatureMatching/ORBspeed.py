"""
Speed-optimized ORB Feature Matching

Properties:
- Fewer features (500) for faster processing
- Larger scale factor (1.5) for fewer pyramid levels
- Minimal pyramid levels (4) to reduce computation
- FLANN-based matcher for faster matching
- Simple top-N filtering without complex verification
- Smaller patch size for faster feature computation
- Reduced edge threshold for faster detection
- Minimal parameter checks and filtering stages

Trade-offs:
+ Significantly faster processing time
+ Lower memory usage
+ Suitable for real-time applications
- May miss some valid matches
- Less robust to scale and viewpoint changes
- Higher chance of false positives
- Less accurate in challenging scenarios
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

def match_orb_features_speed(img1_path, img2_path):
    # Read and convert images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Fast ORB settings
    orb = cv2.ORB_create(
        nfeatures=500,         # Fewer features for speed
        scaleFactor=1.5,       # Larger scale factor = fewer levels
        nlevels=4,             # Fewer pyramid levels
        edgeThreshold=15,      # Smaller edge threshold
        firstLevel=0,
        WTA_K=2,
        patchSize=21,          # Smaller patch size
        fastThreshold=20
    )

    # Detect and compute
    start_time = time.time()
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    # FLANN matcher (faster than BF)
    FLANN_INDEX_LSH = 6
    index_params = dict(
        algorithm=FLANN_INDEX_LSH,
        table_number=4,        # Fewer tables
        key_size=12,
        multi_probe_level=1    # Minimum probe level
    )
    search_params = dict(checks=20)  # Fewer checks for speed
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    matches = flann.match(des1, des2)
    end_time = time.time()

    # Simple distance-based filtering
    matches = sorted(matches, key=lambda x: x.distance)
    good_matches = matches[:30]  # Just take top N matches

    # Visualization
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None,
                                 flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    img_matches = cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(16, 8))
    plt.imshow(img_matches)
    plt.title(f'Speed-optimized ORB matches (Time: {end_time-start_time:.3f}s)')
    plt.axis('off')
    plt.show()

    print(f"Processing time: {end_time-start_time:.3f} seconds")
    print(f"Keypoints in image 1: {len(kp1)}")
    print(f"Keypoints in image 2: {len(kp2)}")
    print(f"Matches found: {len(good_matches)}")

if __name__ == "__main__":
    img1_path = "data/cafe1.jpg"
    img2_path = "data/cafe2.jpg"
    match_orb_features_speed(img1_path, img2_path)