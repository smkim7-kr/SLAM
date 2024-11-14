import cv2
import numpy as np

def add_light_noise(image, light_factor=0.5, noise_factor=25):
    """
    Add light variation and noise to an image
    light_factor: controls brightness (>1 brighter, <1 darker)
    noise_factor: controls the amount of Gaussian noise
    """
    # Convert to float32 for calculations
    img_float = image.astype(np.float32) / 255.0
    
    # Add light variation (multiply)
    img_light = img_float * light_factor
    
    # Add Gaussian noise
    noise = np.random.normal(0, noise_factor/255.0, img_float.shape)
    img_noisy = img_light + noise
    
    # Clip values to valid range [0, 1]
    img_noisy = np.clip(img_noisy, 0, 1)
    
    # Convert back to uint8
    result = (img_noisy * 255).astype(np.uint8)
    
    return result

def main():
    # Read original image
    img = cv2.imread('data/img2.png')
    if img is None:
        raise ValueError("Could not read the image")
    
    # Add light variation and noise
    img_modified = add_light_noise(img, 
                                 light_factor=0.7,    # Make it slightly darker
                                 noise_factor=20)     # Add moderate noise
    
    # Save the modified image
    cv2.imwrite('data/img2_diff.png', img_modified)

if __name__ == "__main__":
    main()