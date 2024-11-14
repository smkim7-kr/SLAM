import cv2
import numpy as np
import matplotlib.pyplot as plt

def adaptive_inverse_projective_mapping_interactive(image_path):
    """
    Performs adaptive inverse projective mapping on the input image.
    Allows the user to interactively select four source points.

    Args:
        image_path (str): Path to the input image.

    Returns:
        None
    """
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to load image at {image_path}")
        return

    # Convert image to RGB (from BGR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Initialize a list to store the selected points
    src_pts = []

    # Function to handle mouse clicks
    def onclick(event):
        if event.xdata is not None and event.ydata is not None:
            if len(src_pts) < 4:
                x, y = event.xdata, event.ydata
                src_pts.append([x, y])
                ax.plot(x, y, 'ro')
                fig.canvas.draw()
                print(f"Point {len(src_pts)}: ({x:.2f}, {y:.2f})")
            if len(src_pts) == 4:
                plt.close()

    # Display the image and set up the click event
    fig, ax = plt.subplots()
    ax.imshow(img_rgb)
    ax.set_title('Click on four points in order (top-left, top-right, bottom-right, bottom-left)')
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    if len(src_pts) < 4:
        print("Error: Four points were not selected.")
        return

    # Convert source points to numpy array
    src_pts = np.float32(src_pts)

    # Image dimensions
    height, width = img.shape[:2]

    # Define destination points for a bird's eye view
    # Estimate the width and height of the warped image based on source points
    src_width = max(
        np.linalg.norm(src_pts[1] - src_pts[0]),  # Top edge
        np.linalg.norm(src_pts[2] - src_pts[3])   # Bottom edge
    )
    src_height = max(
        np.linalg.norm(src_pts[3] - src_pts[0]),  # Left edge
        np.linalg.norm(src_pts[2] - src_pts[1])   # Right edge
    )

    # Define destination points maintaining aspect ratio
    dst_pts = np.float32([
        [0, 0],                    # Top-left
        [src_width - 1, 0],        # Top-right
        [src_width - 1, src_height - 1],  # Bottom-right
        [0, src_height - 1]        # Bottom-left
    ])

    # Compute the perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # Apply the perspective transformation
    warped_img = cv2.warpPerspective(img_rgb, M, (int(src_width), int(src_height)))

    # Visualize the original and warped images
    plt.figure(figsize=(12, 6))

    # Original Image with Source Points
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    # Draw lines connecting the points to show the selected region
    for i in range(4):
        j = (i + 1) % 4
        plt.plot([src_pts[i][0], src_pts[j][0]], 
                [src_pts[i][1], src_pts[j][1]], 'r-')
    plt.scatter(src_pts[:, 0], src_pts[:, 1], c='red', marker='o')
    plt.title('Original Image with Selected Region')
    plt.axis('off')

    # Warped Image (Bird's Eye View)
    plt.subplot(1, 2, 2)
    plt.imshow(warped_img)
    plt.title('Bird\'s Eye View (Inverse Projective Mapping)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    image_path = '../../../KITTI/camera/04/image_0/000004.png' 
    adaptive_inverse_projective_mapping_interactive(image_path)
