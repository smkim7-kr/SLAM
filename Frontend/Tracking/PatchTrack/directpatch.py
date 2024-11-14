import cv2
import numpy as np
import glob
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class Patch:
    def __init__(self, x: int, y: int, size: int = 11, max_iterations: int = 15, precision: float = 0.03):
        self.x = x
        self.y = y
        self.size = size
        self.max_iterations = max_iterations
        self.precision = precision

    def extract(self, image: np.ndarray) -> np.ndarray:
        """Extract patch from image"""
        half = self.size // 2
        
        # Check if patch is within image bounds with a margin
        if (self.x < half or 
            self.x >= image.shape[1] - half or
            self.y < half or 
            self.y >= image.shape[0] - half):
            return np.array([])
        
        # Extract patch only if it's fully within bounds
        patch = image[self.y-half:self.y+half+1, self.x-half:self.x+half+1]
        
        # Double check patch size
        if patch.shape[0] != self.size or patch.shape[1] != self.size:
            return np.array([])
        
        return patch

    def track(self, old_image: np.ndarray, new_image: np.ndarray) -> Tuple[bool, float, float]:
        """Track patch using inverse compositional algorithm"""
        half = self.size // 2
        template = self.extract(old_image)
        
        # Check if template is empty
        if template.size == 0:
            return False, 0, 0
        
        # Compute gradients of template
        grad_x = cv2.Sobel(template, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(template, cv2.CV_32F, 0, 1, ksize=3)
        
        # Pre-compute Jacobian and Hessian
        H = np.zeros((2, 2))
        J_store = np.zeros((self.size, self.size, 2))
        
        for y in range(self.size):
            for x in range(self.size):
                gx = grad_x[y, x]
                gy = grad_y[y, x]
                J = np.array([gx, gy])
                J_store[y, x] = J
                H += np.outer(J, J)
        
        # Initial position
        dx, dy = 0.0, 0.0
        
        for iteration in range(self.max_iterations):
            # Extract patch at current position in new image
            current_x = self.x + int(dx)
            current_y = self.y + int(dy)
            
            # Check if patch is within image bounds
            if (current_x - half < 0 or current_x + half >= new_image.shape[1] or
                current_y - half < 0 or current_y + half >= new_image.shape[0]):
                return False, 0, 0
            
            current = new_image[current_y-half:current_y+half+1, 
                              current_x-half:current_x+half+1].astype(np.float32)
            
            # Verify patch size
            if current.shape != template.shape:
                return False, 0, 0
            
            # Compute error image
            error = template.astype(float) - current
            
            # Add error threshold check
            error_threshold = 30.0
            error_mean = np.mean(np.abs(error))
            if error_mean > error_threshold:
                return False, 0, 0

            # Compute update step
            b = np.zeros(2)
            for y in range(self.size):
                for x in range(self.size):
                    err = error[y, x]
                    J = J_store[y, x]
                    b += J * err
            
            try:
                update = np.linalg.solve(H, b)
            except:
                return False, 0, 0
                
            dx += update[0]
            dy += update[1]

            # Add maximum displacement check
            max_displacement = 30.0  # pixels
            if np.sqrt(dx*dx + dy*dy) > max_displacement:
                return False, 0, 0
            
            # Check convergence
            if np.linalg.norm(update) < self.precision:
                return True, dx, dy
                
        return True, dx, dy

def detect_patches(image: np.ndarray, max_patches: int = 300) -> List[Patch]:
    """Detect patches using Shi-Tomasi corner detector"""
    corners = cv2.goodFeaturesToTrack(
        image,
        maxCorners=max_patches,
        qualityLevel=0.01,
        minDistance=15,
        blockSize=5,
        useHarrisDetector=True,
        k=0.04
    )
    
    if corners is None:
        return []
    
    return [Patch(int(x), int(y)) for x, y in corners.reshape(-1, 2)]

def main():
    # Get images from KITTI dataset
    image_dir = Path("../../../../KITTI/camera/03/image_0")
    image_paths = sorted(glob.glob(str(image_dir / "*.png")))
    
    # Read first frame
    old_frame = cv2.imread(image_paths[0], cv2.IMREAD_GRAYSCALE)
    
    # Detect initial patches
    patches = detect_patches(old_frame)
    
    # Colors for visualization
    POINT_COLOR = (0, 0, 255)   # Red in BGR
    TRACK_COLOR = (0, 255, 0)   # Green in BGR
    
    for image_path in image_paths[1:]:
        frame = cv2.imread(image_path)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Track each patch
        good_patches = []
        flow_vectors = []
        
        for patch in patches:
            success, dx, dy = patch.track(old_frame, frame_gray)
            
            if success:
                good_patches.append(Patch(patch.x + int(dx), patch.y + int(dy)))
                flow_vectors.append((patch.x, patch.y, int(dx), int(dy)))
        
        # Visualization
        for patch in good_patches:
            # Draw current patch center
            cv2.circle(frame, (patch.x, patch.y), 2, POINT_COLOR, -1)
            
            # Draw patch boundary
            half = patch.size // 2
            cv2.rectangle(frame, 
                         (patch.x - half, patch.y - half),
                         (patch.x + half, patch.y + half),
                         POINT_COLOR, 1)
        
        # Draw flow vectors
        for x, y, dx, dy in flow_vectors:
            magnitude = np.sqrt(dx*dx + dy*dy)
            if magnitude > 2.0:  # Only show significant movement
                scale = min(5.0, 30.0/magnitude)  # Limit maximum scale
                scaled_dx = dx * scale
                scaled_dy = dy * scale
                cv2.line(frame, (x, y), (x + int(scaled_dx), y + int(scaled_dy)), TRACK_COLOR, 1)
        
        # Display result
        cv2.imshow('Direct Patch Tracking', frame)
        
        # Break on ESC key
        if cv2.waitKey(30) & 0xFF == 27:
            break
        
        # Update for next iteration
        old_frame = frame_gray
        patches = good_patches
        
        # Add new patches if needed
        if len(patches) < 150:
            new_patches = detect_patches(frame_gray, max_patches=300-len(patches))
            patches.extend(new_patches)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()