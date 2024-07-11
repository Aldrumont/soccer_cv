import cv2
import numpy as np


def nothing(x):
    """
    Callback function for trackbars. Does nothing.
    
    Args:
    x (int): Value from the trackbar (not used).
    """
    pass


# Create a window named 'image'
cv2.namedWindow('image')

# Create 6 trackbars for HSV limits
cv2.createTrackbar('H_low', 'image', 0, 360, nothing)
cv2.createTrackbar('S_low', 'image', 0, 100, nothing)
cv2.createTrackbar('V_low', 'image', 0, 100, nothing)
cv2.createTrackbar('H_high', 'image', 360, 360, nothing)
cv2.createTrackbar('S_high', 'image', 100, 100, nothing)
cv2.createTrackbar('V_high', 'image', 100, 100, nothing)

# Load the image
image = cv2.imread('imgs/ex2.png')

# Resize the image 
height, width, _ = (np.array(image.shape)/1.5).astype(int)
image = cv2.resize(image, (width, height))
# Convert the image to the HSV color space
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

while True:
    # Get the current positions of the trackbars
    h_low = int(cv2.getTrackbarPos('H_low', 'image') / 360 * 255)
    s_low = int(cv2.getTrackbarPos('S_low', 'image') / 100 * 255)
    v_low = int(cv2.getTrackbarPos('V_low', 'image') / 100 * 255)
    h_high = int(cv2.getTrackbarPos('H_high', 'image') / 360 * 255)
    s_high = int(cv2.getTrackbarPos('S_high', 'image') / 100 * 255)
    v_high = int(cv2.getTrackbarPos('V_high', 'image') / 100 * 255)

    # Define the lower and upper HSV limits
    lower = np.array([h_low, s_low, v_low], dtype=np.uint8)
    upper = np.array([h_high, s_high, v_high], dtype=np.uint8)

    # Create a mask with the provided ranges
    mask = cv2.inRange(hsv_image, lower, upper)

    # Apply the mask to the original image
    result = cv2.bitwise_and(image, image, mask=mask)

    # Display the resulting image
    cv2.imshow('image', result)

    # Exit the loop when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Close all windows
cv2.destroyAllWindows()
