import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
import kornia as K
import kornia.feature as KF

def match_lightglue_features(img1_path, img2_path):
    # Read the images in grayscale
    img1 = Image.open(img1_path).convert('RGB')
    img2 = Image.open(img2_path).convert('RGB')
    
    # Resize images if necessary (optional)
    # img1 = img1.resize((640, 480))
    # img2 = img2.resize((640, 480))
    
    # Convert images to tensors and normalize
    img1_t = K.image_to_tensor(np.array(img1), False).float() / 255.0  # Shape: (3, H, W)
    img2_t = K.image_to_tensor(np.array(img2), False).float() / 255.0  # Shape: (3, H, W)
    
    # Add batch dimension and move to device
    img1_t = img1_t.unsqueeze(0)  # Shape: (1, 3, H, W)
    img2_t = img2_t.unsqueeze(0)  # Shape: (1, 3, H, W)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img1_t = img1_t.to(device)
    img2_t = img2_t.to(device)
    
    # Initialize SuperPoint and LightGlue models
    feature_extractor = KF.FeatureExtractor('superpoint', device=device)
    matcher = KF.LightGlue({'weights': 'superpoint'}, device=device)
    
    # Extract features and match
    with torch.inference_mode():
        feats1 = feature_extractor(img1_t)
        feats2 = feature_extractor(img2_t)
        matches = matcher(feats1, feats2)
    
    # Retrieve keypoints and matches
    mkpts0 = matches['keypoints0'][0].cpu().numpy()
    mkpts1 = matches['keypoints1'][0].cpu().numpy()
    mconf = matches['confidence'][0].cpu().numpy()
    
    # Draw matches
    img1_np = np.array(img1)
    img2_np = np.array(img2)
    matched_img = draw_matches(img1_np, img2_np, mkpts0, mkpts1)
    
    # Display results
    plt.figure(figsize=(16, 8))
    plt.imshow(matched_img)
    plt.title('LightGlue Matches')
    plt.axis('off')
    plt.show()
    
    # Print some statistics
    print(f"Total matches found: {len(mkpts0)}")

def draw_matches(img1, img2, kpts1, kpts2):
    # Create a new image by concatenating the two images side by side
    h1, w1, _ = img1.shape
    h2, w2, _ = img2.shape
    max_height = max(h1, h2)
    total_width = w1 + w2
    new_img = np.zeros((max_height, total_width, 3), dtype=img1.dtype)
    new_img[:h1, :w1, :] = img1
    new_img[:h2, w1:w1+w2, :] = img2

    # Adjust keypoint positions for the second image
    kpts2_shifted = kpts2.copy()
    kpts2_shifted[:, 0] += w1

    # Draw lines between matched keypoints
    for pt1, pt2 in zip(kpts1, kpts2_shifted):
        pt1 = tuple(map(int, pt1))
        pt2 = tuple(map(int, pt2))
        color = tuple(np.random.randint(0, 255, 3).tolist())
        cv2.line(new_img, pt1, pt2, color, 1)
        cv2.circle(new_img, pt1, 3, color, -1)
        cv2.circle(new_img, pt2, 3, color, -1)

    return new_img

if __name__ == "__main__":
    img1_path = "data/cafe1.jpg"
    img2_path = "data/cafe2.jpg"
    match_lightglue_features(img1_path, img2_path)
