import cv2
import numpy as np
import matplotlib.pyplot as plt

image_bgr = cv2.imread("photo.jpg", cv2.IMREAD_COLOR)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

plt.figure(figsize=[10, 5])
plt.subplot(121)
plt.imshow(image_bgr)
plt.title("BGR")

plt.subplot(122)
plt.imshow(image_rgb)
plt.title("RGB")

plt.show()

# Brightness
brightness_matrix = np.ones(image_rgb.shape, dtype="uint8") * 50

image_rgb_brighter = cv2.add(image_rgb, brightness_matrix)
image_rgb_darker = cv2.subtract(image_rgb, brightness_matrix)

plt.figure(figsize=[15, 5])
plt.subplot(131)
plt.imshow(image_rgb_brighter)
plt.title("Brighter")

plt.subplot(132)
plt.imshow(image_rgb)
plt.title("Original")

plt.subplot(133)
plt.imshow(image_rgb_darker)
plt.title("Darker")

plt.show()

# Contrast
lower_contrast_matrix = np.ones(image_rgb.shape) * 0.8
higher_contrast_matrix = np.ones(image_rgb.shape) * 1.2

image_rgb_lower_contrast = np.uint8(cv2.multiply(np.float64(image_rgb), lower_contrast_matrix))
# image_rgb_higher_contrast = np.uint8(cv2.multiply(np.float64(image_rgb), higher_contrast_matrix))
image_rgb_higher_contrast = np.uint8(np.clip(cv2.multiply(np.float64(image_rgb), higher_contrast_matrix), 0, 255))

plt.figure(figsize=[15, 5])
plt.subplot(131)
plt.imshow(image_rgb_lower_contrast)
plt.title("Lower Contrast")

plt.subplot(132)
plt.imshow(image_rgb)
plt.title("Original")

plt.subplot(133)
plt.imshow(image_rgb_higher_contrast)
plt.title("Higher Contrast")

plt.show()

# Mask
mask = cv2.imread("mask.png", cv2.IMREAD_GRAYSCALE)
mask_inv = cv2.bitwise_not(mask)

plt.figure(figsize=[10, 5])
plt.subplot(121)
plt.imshow(image_rgb)
plt.title("Original")

plt.subplot(122)
plt.imshow(mask, cmap="gray")
plt.title("Mask")

plt.show()

colorful = cv2.imread("colorful.jpg", cv2.IMREAD_COLOR)
colorful_rgb = cv2.cvtColor(colorful, cv2.COLOR_BGR2RGB)

print(colorful_rgb.shape)

colorful_rgb = cv2.resize(colorful_rgb, mask.shape[::-1], interpolation=cv2.INTER_AREA)

print(mask.shape)
print(colorful_rgb.shape)

plt.figure(figsize=[10, 5])
plt.subplot(121)
plt.imshow(colorful_rgb)
plt.title("Original")

plt.subplot(122)
plt.imshow(mask, cmap="gray")
plt.title("Mask")

plt.show()

colorful_and = cv2.bitwise_and(colorful_rgb, colorful_rgb, mask=mask)
image_and = cv2.bitwise_and(image_rgb, image_rgb, mask=mask_inv)

merged_image = cv2.add(colorful_and, image_and)
plt.imshow(merged_image)
plt.legend("Merged Image")
plt.show()

cv2.imshow("Merged", merged_image)
cv2.waitKey(0)