import cv2
import numpy as np

def read_calib_file(calib_path):
    """Read KITTI calibration file."""
    with open(calib_path, 'r') as f:
        calib = {}
        for line in f.readlines():
            if len(line) == 0: continue
            key, value = line.split(':', 1)
            calib[key] = np.array([float(x) for x in value.split()])
            calib[key] = calib[key].reshape(3, 4) if len(calib[key]) == 12 else calib[key]
    return calib

def stitch_images():
    # Paths
    img1_path = 'data/img1.png'
    img2_path = 'data/img2.png'

    # Read images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    if img1 is None or img2 is None:
        raise ValueError("Could not read the images")

    # Convert images to grayscale for feature detection
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Find keypoints and descriptors
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    # FLANN matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    # Apply Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # Get matched keypoints
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Calculate size of the stitched image
    height, width = img1.shape[:2]
    # Create a transformation matrix to shift the first image
    offset = np.array([[1, 0, width], [0, 1, 0], [0, 0, 1]])
    # Combine the offset with the homography
    H_offset = offset @ H

    # Create panorama with doubled width
    stitched = np.zeros((height, width*2, 3), dtype=np.uint8)
    
    # Place the second image on the left
    stitched[0:height, 0:width] = img2
    
    # Warp and place the first image
    warped = cv2.warpPerspective(img1, H_offset, (width*2, height))
    
    # Blend the images where they overlap
    mask = (warped != 0)  # Create a mask where warped image has non-zero values
    stitched[mask] = warped[mask]

    # Save result
    cv2.imwrite('stitched_result.png', stitched)

if __name__ == "__main__":
    stitch_images()
