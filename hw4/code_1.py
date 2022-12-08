import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import signal
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

def rgb_to_hsi(img):
    rows = int(img.shape[0])
    cols = int(img.shape[1])
    R, G, B = cv2.split(img)
    R = R / 255.0
    G = G / 255.0
    B = B / 255.0
    dst = img.copy()
    H, S, I = cv2.split(dst)
    for i in range(rows):
        for j in range(cols):
            num = 0.5 * ((R[i, j] - G[i, j]) + (R[i, j] - B[i, j]))
            den = np.sqrt((R[i, j] - G[i, j]) ** 2 + (R[i, j] - B[i, j]) * (G[i, j] - B[i, j]))

            if den == 0:
                H = 0
            else:
                theta = float(np.arccos(num / den))
                if B[i, j] <= G[i, j]:
                    H = theta
                else:
                    H = 2 * math.pi - theta

            min_rgb = min(min(R[i, j], G[i, j]), B[i, j])
            sum = R[i, j] + G[i, j] + B[i, j]
            if sum == 0:
                S = 0
            else:
                S = 1 - 3 * min_rgb / sum
            
            H = H / (2 * math.pi)
            I = sum / 3.0
            dst[i, j, 0] = H * 255
            dst[i, j, 1] = S * 255
            dst[i, j, 2] = I * 255
    return dst

def hsi_to_rgb(img):
    rows = int(img.shape[0])
    cols = int(img.shape[1])
    H, S, I = cv2.split(img)
    H = H / 255.0
    S = S / 255.0
    I = I / 255.0
    dst = img.copy()
    B, G, R = cv2.split(dst)
    for i in range(rows):
        for j in range(cols):
            H[i, j] *= 360
            if H[i, j] >= 0 and H[i, j] < 120:
                B = I[i, j] * (1 - S[i, j])
                R = I[i, j] * (1 + (S[i, j] * math.cos(H[i, j]*math.pi/180)) / math.cos((60 - H[i, j])*math.pi/180))
                G = 3 * I[i, j] - (R + B)
            elif H[i, j] >= 120 and H[i, j] < 240:
                H[i, j] = H[i, j] - 120
                R = I[i, j] * (1 - S[i, j])
                G = I[i, j] * (1 + (S[i, j] * math.cos(H[i, j]*math.pi/180)) / math.cos((60 - H[i, j])*math.pi/180))
                B = 3 * I[i, j] - (R + G)
            elif H[i, j] >= 240 and H[i, j] <= 360:
                H[i, j] = H[i, j] - 240
                G = I[i, j] * (1 - S[i, j])
                B = I[i, j] * (1 + (S[i, j] * math.cos(H[i, j]*math.pi/180)) / math.cos((60 - H[i, j])*math.pi/180))
                R = 3 * I[i, j] - (G + B)
            dst[i, j, 0] = R * 255
            dst[i, j, 1] = G * 255
            dst[i, j, 2] = B * 255
    return dst
if __name__ == '__main__':
    img = cv2.imread('LovePeace rose.tif') # fruit blurred-noisy.tif index.jpeg
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    path = 'Result/'

    # (a):
    red = img.copy()
    green = img.copy()
    blue = img.copy()
    red[:, :, 1] = 0
    red[:, :, 2] = 0
    green[:, :, 0] = 0
    green[:, :, 2] = 0
    blue[:, :, 0] = 0
    blue[:, :, 1] = 0
    red = cv2.cvtColor(red, cv2.COLOR_RGB2GRAY)
    green = cv2.cvtColor(green, cv2.COLOR_RGB2GRAY)
    blue = cv2.cvtColor(blue, cv2.COLOR_RGB2GRAY)
    save_img(path, 'red', red)
    save_img(path, 'green', green)
    save_img(path, 'blue', blue)

    hsi = rgb_to_hsi(img)
    h = hsi.copy()
    s = hsi.copy()
    i = hsi.copy()
    h[:, :, 1] = 0
    h[:, :, 2] = 0
    s[:, :, 0] = 0
    s[:, :, 2] = 0
    i[:, :, 0] = 0
    i[:, :, 1] = 0
    h = cv2.cvtColor(h, cv2.COLOR_RGB2GRAY)
    s = cv2.cvtColor(s, cv2.COLOR_RGB2GRAY)
    i = cv2.cvtColor(i, cv2.COLOR_RGB2GRAY)
    save_img(path, 'h', h)
    save_img(path, 's', s)
    save_img(path, 'i', i)

    # (b):
    kernel = np.array([[-1, -1, -1],
                        [-1, 9,-1],
                        [-1, -1, -1]])
    sharpended_rgb = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)
    sharpended_hsi = cv2.filter2D(src=hsi, ddepth=-1, kernel=kernel)
    sharpended_hsi = hsi_to_rgb(sharpended_hsi)
    save_img(path, 'sharpened_rgb', sharpended_rgb)
    save_img(path, 'sharpended_hsi', sharpended_hsi)

    # (c):
    diff = sharpended_rgb - sharpended_hsi
    diff = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
    save_img(path, 'diff', diff)
    

