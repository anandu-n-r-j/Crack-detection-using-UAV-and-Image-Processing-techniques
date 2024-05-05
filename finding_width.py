import cv2
import matplotlib.pyplot as plt
import numpy as np

# Loading the image again
image = cv2.imread(r'C:\Users\anand\OneDrive\Documents\BTP-Crack Detection\pred_028.jpg', 0)

# Improving the image by ignoring small white pixels that may not be part of the crack
# Applying a morphological opening to remove small objects
kernel = np.ones((3, 3), np.uint8)
opened_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=2)

# After opening, we need to re-identify the crack by finding contours again
contours, _ = cv2.findContours(opened_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filtering out contours that are too small which are likely not cracks
filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 100]

# Creating an empty list to store widths
widths = []

# Creating an empty image to draw the contours and bounding boxes
crack_image = np.zeros_like(image)

# Drawing the contours and minimum bounding rectangles around each contour
for contour in filtered_contours:
    # Geting the minimum bounding rectangle
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    
    # Drawing the contour and bounding box
    cv2.drawContours(crack_image, [contour], 0, (255, 255, 255), 1)
    cv2.drawContours(crack_image, [box], 0, (255, 255, 255), 1)
    
    # Calculating width of the bounding box
    width = np.linalg.norm(box[0] - box[1])  # Euclidean distance between two points
    # Appending width to the list
    widths.append(width)

# Calculating average width
if widths:  # Checking if list is not empty
    average_width = sum(widths) / len(widths)
    print(f"The average width is {average_width:.1f} pixels")

# Ploting the image with crack contours and bounding boxes
plt.figure(figsize=(10, 10))
plt.imshow(crack_image, cmap='gray')
plt.title('Crack Contours with Minimum Bounding Rectangles')
plt.show()
