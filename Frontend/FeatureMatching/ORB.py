import cv2
import numpy as np
import matplotlib.pyplot as plt

def match_orb_features(img1_path, img2_path):
    # Read the images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    # Convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Initialize ORB detector
    orb = cv2.ORB_create(nfeatures=2000)

    # Find keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    # Create BF Matcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors
    matches = bf.match(des1, des2)

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Add distance threshold for good matches
    distance_threshold = 40  # Adjust this value based on your needs
    
    # Prepare lists for good and bad matches
    good_matches = [m for m in matches if m.distance < distance_threshold]
    bad_matches = [m for m in matches if m.distance >= distance_threshold]

    # Create empty image for all matches
    img_matches = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], 3), dtype=np.uint8)
    
    # Draw matches with different colors and opacity
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None,
                                 matchColor=(0, 255, 0),  # Green for good matches
                                 flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    # Create a copy for the transparent lines
    overlay = img_matches.copy()
    overlay = cv2.drawMatches(img1, kp1, img2, kp2, bad_matches, overlay,
                             matchColor=(0, 0, 255),  # Red for bad matches
                             flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS | 
                                   cv2.DrawMatchesFlags_DRAW_OVER_OUTIMG)
    
    # Blend the overlay with the original image
    alpha = 0.3
    img_matches = cv2.addWeighted(overlay, alpha, img_matches, 1 - alpha, 0)

    # Convert BGR to RGB for matplotlib
    img_matches = cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB)

    # Display results with updated title
    plt.figure(figsize=(16, 8))
    plt.imshow(img_matches)
    plt.title(f'ORB matches (threshold={distance_threshold})\n'
              f'Good matches (green): {len(good_matches)}, Bad matches (red): {len(bad_matches)}')
    plt.axis('off')
    plt.show()

    # Update statistics printing
    print(f"Total keypoints in image 1: {len(kp1)}")
    print(f"Total keypoints in image 2: {len(kp2)}")
    print(f"Good matches (distance < {distance_threshold}): {len(good_matches)}")
    print(f"Bad matches (distance ≥ {distance_threshold}): {len(bad_matches)}")

if __name__ == "__main__":
    img1_path = "data/cafe1.jpg"
    img2_path = "data/cafe2.jpg"
    match_orb_features(img1_path, img2_path)