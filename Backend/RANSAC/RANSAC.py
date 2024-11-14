import cv2
import numpy as np
import time
from pathlib import Path
from collections import defaultdict
import pandas as pd

class ImageMatcher:
    def __init__(self):
        self.detector = cv2.SIFT_create()
        self.matcher = cv2.BFMatcher_create(cv2.NORM_L2, crossCheck=True)
        
    def detect_and_match(self, img1, img2):
        kp1, des1 = self.detector.detectAndCompute(img1, None)
        kp2, des2 = self.detector.detectAndCompute(img2, None)
        matches = self.matcher.match(des1, des2)
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        return pts1, pts2, matches, kp1, kp2

def visualize_matches(img1, img2, kp1, kp2, matches, mask=None, title="Matches"):
    """Visualize matches between two images"""
    if mask is not None:
        # Keep only inlier matches
        matches = [m for m, msk in zip(matches, mask) if msk]
    
    # Draw matches
    img_matches = cv2.drawMatches(
        img1, kp1, img2, kp2, matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    
    # Add title and match count
    text = f"{title} - {len(matches)} matches"
    cv2.putText(img_matches, text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return img_matches

def run_ransac_experiment(pts1, pts2, method, reproj_threshold, confidence):
    """Run RANSAC with different methods and parameters"""
    start_time = time.time()
    
    if method == "RANSAC":
        F, mask = cv2.findFundamentalMat(pts1, pts2, 
                                        cv2.FM_RANSAC, 
                                        reproj_threshold, 
                                        confidence)
    elif method == "PROSAC":
        F, mask = cv2.findFundamentalMat(pts1, pts2, 
                                        cv2.FM_RANSAC + cv2.USAC_PROSAC, 
                                        reproj_threshold, 
                                        confidence)
    elif method == "LO-RANSAC":
        F, mask = cv2.findFundamentalMat(pts1, pts2, 
                                        cv2.FM_RANSAC + cv2.USAC_ACCURATE, 
                                        reproj_threshold, 
                                        confidence)
    elif method == "MAGSAC":
        F, mask = cv2.findFundamentalMat(pts1, pts2, 
                                        cv2.USAC_MAGSAC, 
                                        reproj_threshold, 
                                        confidence)
    
    end_time = time.time()
    
    if mask is None:
        return None, None, 0, np.inf
    
    inlier_ratio = np.sum(mask) / len(mask)
    computation_time = end_time - start_time
    
    return F, mask, inlier_ratio, computation_time

def main():
    # Read images
    img1 = cv2.imread('data/img1.png')
    img2 = cv2.imread('data/img2_diff.png')
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Initialize matcher
    matcher = ImageMatcher()
    
    # Parameters to test
    methods = ["RANSAC", "PROSAC", "LO-RANSAC", "MAGSAC"]
    reproj_thresholds = [1.0, 2.0, 3.0]
    confidences = [0.95, 0.99]
    
    # Store results
    results = []
    best_visualization = None
    best_score = -1
    
    print("Running experiments...")
    
    # Run experiments
    for i in range(100):  # 100 trials
        if i % 10 == 0:
            print(f"Trial {i}/100")
            
        # Get matches for this trial
        pts1, pts2, matches, kp1, kp2 = matcher.detect_and_match(img1_gray, img2_gray)
        
        for method in methods:
            for thresh in reproj_thresholds:
                for conf in confidences:
                    F, mask, inlier_ratio, comp_time = run_ransac_experiment(
                        pts1, pts2, method, thresh, conf
                    )
                    
                    if mask is not None:
                        results.append({
                            'method': method,
                            'reproj_threshold': thresh,
                            'confidence': conf,
                            'inlier_ratio': inlier_ratio,
                            'computation_time': comp_time
                        })
                        
                        # Update best visualization if this is the best result so far
                        current_score = inlier_ratio / comp_time
                        if current_score > best_score:
                            best_score = current_score
                            best_visualization = visualize_matches(
                                img1, img2, kp1, kp2, matches, mask,
                                f"{method} (thresh={thresh}, conf={conf})"
                            )
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Group by method and parameters, calculate mean performance
    summary = df.groupby(['method', 'reproj_threshold', 'confidence']).agg({
        'inlier_ratio': ['mean', 'std'],
        'computation_time': ['mean', 'std']
    }).reset_index()
    
    # Normalize metrics and find best method
    normalized_summary = summary.copy()
    normalized_summary['inlier_ratio_mean_norm'] = (summary[('inlier_ratio', 'mean')] - 
                                                  summary[('inlier_ratio', 'mean')].min()) / \
                                                 (summary[('inlier_ratio', 'mean')].max() - 
                                                  summary[('inlier_ratio', 'mean')].min())
    
    normalized_summary['computation_time_mean_norm'] = 1 - (summary[('computation_time', 'mean')] - 
                                                          summary[('computation_time', 'mean')].min()) / \
                                                         (summary[('computation_time', 'mean')].max() - 
                                                          summary[('computation_time', 'mean')].min())
    
    normalized_summary['score'] = (normalized_summary['inlier_ratio_mean_norm'] + 
                                 normalized_summary['computation_time_mean_norm']) / 2
    
    # Get best method
    best_result = normalized_summary.loc[normalized_summary['score'].idxmax()]
    
    print("\nBest Method:")
    print(f"Method: {best_result['method']}")
    print(f"Reprojection Threshold: {best_result['reproj_threshold']}")
    print(f"Confidence: {best_result['confidence']}")
    print(f"Average Inlier Ratio: {best_result[('inlier_ratio', 'mean')]:.3f}")
    print(f"Average Computation Time: {best_result[('computation_time', 'mean')]:.3f} seconds")
    
    # Save results
    summary.to_csv('ransac_comparison_results.csv')
    
    # Show and save best visualization
    if best_visualization is not None:
        cv2.imshow('Best Match Result', best_visualization)
        cv2.imwrite('best_matches.png', best_visualization)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()