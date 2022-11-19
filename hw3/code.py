import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import signal
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

def handel_histogram(path, img):
    file = open(path + 'value.txt', 'w+')
    for i in range(256):
        file.write(str(np.count_nonzero(img == i)) + '\n')
    plt.hist(img.ravel(), bins=256)
    plt.savefig(path + 'hist.tif', dpi=200)
    plt.clf()

def alpha_trimmed_mean_filter(image, size = 5, alpha = 16):
    output = np.zeros(image.shape, np.uint8)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            temp = []
            sum = 0
            for x in range(-2, 3):
                for y in range(-2, 3):
                    gray = 0
                    if i+x >= 0 and i+x < image.shape[0] and j+y >= 0 and j+y < image.shape[1]:
                        gray = image[i+x, j+y]
                    temp.append(gray)
            temp.sort()
            for k in range(alpha // 2, size**2 - (alpha // 2)):
                k -= 1
                sum += temp[k]
            output[i][j] = sum / (size**2 - alpha)
    return output

if __name__ == '__main__':
    img = cv2.imread('Kid2 degraded.tiff', cv2.IMREAD_GRAYSCALE) # fruit blurred-noisy.tif index.jpeg
    path = 'Result/'
    rows, cols = img.shape
    print(rows, cols)
    rows_double = rows * 2
    cols_double = cols * 2
    
    # (a): Original
    save_img(path, '1', img)

    # (b):
    trim_img = img[710:760, 150:200]
    save_img(path, '2', trim_img)
    handel_histogram(path, trim_img)

    # (c):
    alpha_img = alpha_trimmed_mean_filter(img)
    save_img(path, '3', alpha_img)

    # (d):
    for D0 in range(100, 300, 50):
        for n in range(3):
            n = n + 1
            blpf = np.zeros((rows_double, cols_double), dtype=np.float32)
            for u in range(rows_double):
                for v in range(cols_double):
                    temp = np.sqrt((u-rows_double/2)**2 + (v-cols_double/2)**2)
                    blpf[u, v] = 1 / (1 + (temp / D0)**(2*n))

            glpf = np.zeros((rows_double, cols_double), dtype=np.float32)
            for u in range(rows_double):
                for v in range(cols_double):
                    temp = np.sqrt((u-rows_double/2)**2 + (v-cols_double/2)**2)
                    glpf[u, v] = np.exp(-temp**2/(2*D0*D0))

            img_pad = cv2.copyMakeBorder(alpha_img, 0, rows, 0, cols, cv2.BORDER_CONSTANT, None, value = 0)
            img_pad_f = np.fft.fft2(np.float32(img_pad))
            img_pad_fshift = np.fft.fftshift(img_pad_f)
            low_pass = img_pad_fshift * blpf
            low_pass = low_pass * glpf
            low_pass = np.fft.ifftshift(low_pass)
            low_result = np.abs(np.fft.ifft2(low_pass))
            low_result = low_result[0:rows, 0:cols]
            save_img(path, str(D0) + '-' + str(n), low_result)