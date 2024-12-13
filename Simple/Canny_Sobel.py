# import cv2
# for i in range(1,33):
#     image = cv2.imread(f'../train/images/test{i}.jpg')

#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     edges = cv2.Canny(gray, 90, 100)

#     contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

#     # Display the result
#     cv2.namedWindow('Contours', cv2.WINDOW_NORMAL)
#     cv2.imshow('Contours', image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

import cv2, numpy as np

for i in range(1,33):
    
    image = cv2.imread(f'../../train/images/test{i}.jpg')


    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=9)  # Gradient in x direction
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=9)  # Gradient in y direction

    magnitude = cv2.magnitude(grad_x, grad_y)

    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

    magnitude = np.uint8(magnitude)

    _, binary_image = cv2.threshold(magnitude, 64, 255, cv2.THRESH_BINARY)  # Adjust threshold value as needed


    kernel = np.ones((3, 3), np.uint8)
    dilated_edges = cv2.dilate(binary_image, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    cv2.drawContours(image, contours, -1, (0, 255, 0), 2)  # Green contours with thickness 2

    # Display the result
    cv2.namedWindow('Contours', cv2.WINDOW_NORMAL)
    cv2.imshow('Contours', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
