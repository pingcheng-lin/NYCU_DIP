from shutil import ExecError
from xml.dom.pulldom import END_ELEMENT
import cv2
from cv2 import CV_16S
import numpy as np
import matplotlib.pyplot as plt
if __name__ == '__main__':
    img = cv2.imread('kid blurred-noisy.tif', cv2.IMREAD_GRAYSCALE) #    fruit blurred-noisy.tif index.jpeg

    # (b): Laplacian
    lap = cv2.Laplacian(img, cv2.CV_32F, ksize=1)
    max = np.amax(lap)
    min = np.amin(lap)
    print(max)
    print(min)
    copy = lap.copy()
    rows, cols = copy.shape
    print(rows)
    print(cols)
    for i in range(rows):
        for j in range(cols):
            copy[i, j] = (copy[i, j]+128)
            # print(copy[i, j])
    
    cv2.imwrite('Result/laplacian.jpg', copy)

    # (c): Laplacian-sharpened
    temp1 = np.asarray(img, np.float64)
    temp2 = np.asarray(lap, np.float64)
    sharpended_img = cv2.add(temp1, temp2)

    cv2.imwrite('Result/laplacian-sharpened.jpg', sharpended_img)
    exit()
    # cv2.imshow('My image', sharpended_img)
    # cv2.waitKey(0)

    # (d): Sobel gradient
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    sobel_mix = cv2.add(sobelx, sobely)

    cv2.imshow('My image', sobel_mix)
    cv2.waitKey(0)

    # (e): Smoothed gradient
    kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]])
    smoothed_img = cv2.filter2D(sobel_mix, ddepth=cv2.CV_64F, kernel=kernel)

    cv2.imshow('My image', smoothed_img)
    cv2.waitKey(0)

    # (f): (e)x(b)
    multiply_result = cv2.multiply(lap, smoothed_img)
    cv2.imshow('My image', multiply_result)
    cv2.waitKey(0)


    # (g): (a)+(f)
    add_result = img + multiply_result

    cv2.imshow('My image', add_result)
    cv2.waitKey(0)

    # (h): Power-law transformation
    gamma = 0.5
    final_img = np.array(255*(img/255)**0.4, dtype='uint8')

    cv2.imshow('My image', final_img)
    cv2.waitKey(0)





    # rows, cols = lap.shape
    # print(rows)
    # print(cols)
    # for i in range(rows):
    #     for j in range(cols):
    #         level = lap[i, j]
    #         # lap[i, j] = lap[i, j] + 128
    #         if level > 255:
    #             lap[i, j] = 255
    #         elif level < 0:
    #             lap[i, j] = -lap[i, j]
    #             if lap[i, j] > 255:
    #                 lap[i, j] = 255

    # lap2 = cv2.Laplacian(img, cv2.CV_64F, ksize=5)
    # kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    # lap = cv2.filter2D(img, ddepth=cv2.CV_64F, kernel=kernel)

    # dst = cv2.add(original_img, lap)
    # dst = cv2.convertScaleAbs(lap)


    # cv2.imwrite("Preview.jpg", lap)
    cv2.destroyAllWindows()

    # plt.imshow(lap, cmap='gray')
    # plt.show()