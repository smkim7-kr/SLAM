import cv2
import numpy as np
import glob
from pathlib import Path

def draw_flow_vectors(img, flow, step=16, scale=1, color=(0, 255, 0)):
    """Draw flow vectors on the image"""
    h, w = img.shape[:2]
    y, x = np.mgrid[step//2:h:step, step//2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    
    # Create line endpoints
    lines = np.vstack([x, y, x+fx*scale, y+fy*scale]).T.reshape(-1, 2, 2)
    lines = np.int32(lines)
    
    # Create visualization image
    vis = img.copy()
    for (x1, y1), (x2, y2) in lines:
        cv2.arrowedLine(vis, (x1, y1), (x2, y2), color, 1, tipLength=0.2)
    return vis

def main(vis_type='vector'):
    # Get first 300 images from KITTI dataset
    image_dir = Path("../../../../KITTI/camera/03/image_0")
    image_paths = sorted(glob.glob(str(image_dir / "*.png")))[:300]
    
    # Read first frame
    old_frame = cv2.imread(image_paths[0])
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    
    # Parameters for Farneback optical flow
    flow_params = dict(
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )
    
    # Create HSV mask for visualization if using HSV mode
    if vis_type == 'hsv':
        hsv = np.zeros_like(old_frame)
        hsv[..., 1] = 255
    
    for image_path in image_paths[1:]:
        # Read current frame
        frame = cv2.imread(image_path)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(
            old_gray,
            gray,
            None,
            **flow_params
        )
        
        # Visualization
        if vis_type == 'hsv':
            # HSV visualization
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[..., 0] = angle * 180 / np.pi / 2
            hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
            vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            
        else:  # vector visualization
            # Draw flow vectors
            vis = draw_flow_vectors(frame, flow, 
                                  step=16,        # Grid size for vectors
                                  scale=3,        # Scale factor for vector length
                                  color=(0, 255, 0))  # Green color for vectors
        
        # Display the results
        window_name = 'Dense Optical Flow - ' + vis_type.capitalize()
        cv2.imshow(window_name, vis)
        
        # Break on ESC key
        if cv2.waitKey(30) & 0xFF == 27:
            break
        
        # Update previous frame
        old_gray = gray.copy()
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Choose visualization type: 'vector' or 'hsv'
    main(vis_type='vector')  # Change this to 'hsv' for HSV visualization
