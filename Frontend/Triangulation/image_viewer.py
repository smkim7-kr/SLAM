import rerun as rr
import cv2
import numpy as np
import os

# Read the images
img1 = cv2.imread("../IPM/data/img1.png")
img2 = cv2.imread("../IPM/data/img2.png")

# Convert from BGR to RGB (rerun expects RGB)
img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

# Initialize rerun with recording name
rr.init("image_viewer", spawn=True)

# Log the images to rerun in 3D space
# First image at origin
rr.log("world/image1", 
    rr.Image(img1_rgb),
    rr.Transform3D(translation=[0, 0, 0])  # Position at origin
)

# Second image offset in space (for example, 2 units to the right)
rr.log("world/image2", 
    rr.Image(img2_rgb),
    rr.Transform3D(translation=[2, 0, 0])  # Position 2 units along x-axis
)
