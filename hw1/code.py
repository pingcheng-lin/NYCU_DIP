import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def save_img(path, name, img):
    temp = Image.fromarray(img).convert('RGB')
    temp.save(path + name + '.tif', dpi = (200, 200))

def handel_histogram(path, type, img):
    file = open(path + type + '_value.txt', 'w+')
    for i in range(256):
        file.write(str(np.count_nonzero(img == i)) + '\n')
    plt.hist(img.ravel(), bins=256)
    plt.savefig(path + type + '_hist.tif')
    plt.clf()

def contrast_stretching(img, rows, cols):
    max = np.amax(img)
    min = np.amin(img)
    for i in range(rows):   
        for j in range(cols):
            img[i, j] = ((img[i, j] - min) / (max - min)) * 255
    return img

if __name__ == '__main__':
    for i in ['kid', 'fruit']:
        target = 'kid blurred-noisy.tif'
        if i == 'fruit':
            target = 'fruit blurred-noisy.tif'
        img = cv2.imread(target, cv2.IMREAD_GRAYSCALE) # fruit blurred-noisy.tif index.jpeg
        rows, cols = img.shape
        path = 'Result/' + i + '/'

        # (a): Original
        save_img(path, '1.original', img)

        # (b): Laplacian
        lap = cv2.Laplacian(img, cv2.CV_32F, ksize = 9)
        lap_copy = lap.copy()
        lap_copy = np.abs(lap_copy)
        lap_copy = contrast_stretching(lap_copy, rows, cols)
        save_img(path, '2.laplacian', lap_copy)

        # (c): Laplacian-sharpened
        temp1 = np.asarray(img, np.float64)
        temp2 = np.asarray(lap_copy, np.float64)
        sharpended_img = cv2.add(temp1, temp2)
        copy = sharpended_img.copy()
        copy = np.abs(copy)
        copy = contrast_stretching(copy, rows, cols)
        save_img(path, '3.laplacian-sharpened', copy)

        # (d): Sobel gradient
        sobelx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
        sobely = cv2.Sobel(img, cv2.CV_32F, 0, 1)
        sobel_mix = cv2.add(np.abs(sobelx), np.abs(sobely))
        sobel_mix = contrast_stretching(sobel_mix, rows, cols)
        save_img(path, '4.sobel', sobel_mix)

        # (e): Smoothed gradient
        kernel = np.ones((5, 5), np.float32) / 25
        smoothed_img = cv2.filter2D(sobel_mix, -1, kernel=kernel)
        smoothed_img = contrast_stretching(smoothed_img, rows, cols)
        save_img(path, '5.smooth', smoothed_img)

        # (f): (e)x(b)
        temp1 = np.asarray(lap_copy, np.float64)
        temp2 = np.asarray(smoothed_img, np.float64)
        multiply_result = cv2.multiply(temp1, temp2)
        multiply_result = contrast_stretching(multiply_result, rows, cols)
        save_img(path, '6.multiply', multiply_result)

        # (g): (a)+(f)
        temp1 = np.asarray(img, np.float64)
        add_result = cv2.addWeighted(temp1, 0.6, temp2, 0.4, 0)
        add_result = contrast_stretching(add_result, rows, cols)
        save_img(path, '7.add', add_result)

        # (h): Power-law transformation
        gamma = 0.8
        final_img = np.array(255*(np.abs(add_result)/255)**gamma, dtype='uint8')
        final_img = contrast_stretching(final_img, rows, cols)
        save_img(path, '8.final', final_img)

        # Original histogram
        handel_histogram(path, '9.original', img)

        # Output histogram
        handel_histogram(path, '10.output', final_img)