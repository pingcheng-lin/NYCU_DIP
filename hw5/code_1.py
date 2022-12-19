import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage, signal
import math
def save_img(path, name, img):
    fig = plt.figure()
    fig.set_size_inches((4,4))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.set_cmap('gray')
    ax.imshow(img)
    plt.savefig(path + name + '.tif', dpi=200)
    plt.close()

def non_max_suppression(magnitude, angle):
    rows, cols = magnitude.shape
    output = np.zeros(magnitude.shape)
    PI = 180

    for row in range(1, rows - 1):
        for col in range(1, cols - 1):
            angle_val = angle[row, col]

            if (0 <= angle_val < PI / 8) or (15 * PI / 8 <= angle_val <= 2 * PI):
                before_pixel = magnitude[row, col - 1]
                after_pixel = magnitude[row, col + 1]
            elif (PI / 8 <= angle_val < 3 * PI / 8) or (9 * PI / 8 <= angle_val < 11 * PI / 8):
                before_pixel = magnitude[row + 1, col - 1]
                after_pixel = magnitude[row - 1, col + 1]
            elif (3 * PI / 8 <= angle_val < 5 * PI / 8) or (11 * PI / 8 <= angle_val < 13 * PI / 8):
                before_pixel = magnitude[row - 1, col]
                after_pixel = magnitude[row + 1, col]
            else:
                before_pixel = magnitude[row - 1, col - 1]
                after_pixel = magnitude[row + 1, col + 1]

            if magnitude[row, col] >= before_pixel and magnitude[row, col] >= after_pixel:
                output[row, col] = magnitude[row, col]

    return output

def threshold(img):
    highThreshold = 255*0.1
    lowThreshold = 255*0.04

    M, N = img.shape
    res = np.zeros((M,N), dtype=np.int32)
    nh = res.copy()
    nl = res.copy()
    
    for i in range(1, M-1):
        for j in range(1, N-1):
            if img[i, j] >= highThreshold:
                res[i, j] = img[i, j]
                nh[i, j] = img[i, j]
            elif (img[i, j] <= highThreshold) and (img[i, j] >= lowThreshold):
                res[i, j] = img[i, j]
                nl[i, j] = img[i, j]
    
    return (res, nh, nl)

def hysteresis(img):
    highThreshold = 255*0.1
    lowThreshold = 255*0.04
    M, N = img.shape  
    for i in range(1, M-1):
        for j in range(1, N-1):
            if (img[i, j] <= highThreshold) and (img[i, j] >= lowThreshold):
                if ((img[i+1, j-1] >= highThreshold) or (img[i+1, j] >= highThreshold) or (img[i+1, j+1] >= highThreshold)
                    or (img[i, j-1] >= highThreshold) or (img[i, j+1] >= highThreshold)
                    or (img[i-1, j-1] >= highThreshold) or (img[i-1, j] >= highThreshold) or (img[i-1, j+1] >= highThreshold)):
                    img[i, j] = 255
                else:
                    img[i, j] = 0
            elif img[i, j] >= highThreshold:
                img[i, j] = 255
    return img

if __name__ == '__main__':
    img = cv2.imread('Kid at playground.tif') # fruit blurred-noisy.tif index.jpeg
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    path = 'Result/'

    # (a):
    blur = cv2.GaussianBlur(img, (29, 29), int(img.shape[0]) * 0.005)
    save_img(path, 'blur', blur)
    
    gX = cv2.Sobel(blur, cv2.CV_64F, 1, 0)
    gY = cv2.Sobel(blur, cv2.CV_64F, 0, 1)
    magnitude = np.sqrt((gX ** 2) + (gY ** 2))
    angle = np.arctan2(gY, gX) * (180 / np.pi) % 180
    save_img(path, 'magnitude', magnitude)
    save_img(path, 'angle', angle)

    # (b):
    sup = non_max_suppression(magnitude, angle)
    save_img(path, 'sup', sup)
    thres, nh, nl = threshold(sup)
    save_img(path, 'nh', nh)
    save_img(path, 'nl', nl)
    result = hysteresis(thres)
    save_img(path, 'result', result)


