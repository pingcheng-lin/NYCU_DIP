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
    plt.savefig(path + name + '.tif', dpi=150)
    plt.close()

if __name__ == '__main__':
    for i in ['kid', 'fruit']:
        img = cv2.imread(i + '.tif', cv2.IMREAD_GRAYSCALE) # fruit blurred-noisy.tif index.jpeg
        path = 'Result/' + i + '/'
        rows, cols = img.shape
        rows_double = rows * 2
        cols_double = cols * 2
        
        # (a): Original
        save_img(path, '1', img)

        # (b):
        f = np.fft.fft2(np.float32(img))
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20*np.log(np.abs(fshift))
        save_img(path, '2', magnitude_spectrum)

        # (c):
        lpf = np.zeros((rows_double, cols_double), dtype=np.float32)
        D0 = 200
        for u in range(rows_double):
            for v in range(cols_double):
                temp = np.sqrt((u-rows_double/2)**2 + (v-cols_double/2)**2)
                lpf[u, v] = np.exp(-temp**2/(2*D0*D0))
        hpf = 1 - lpf
        save_img(path, '3', lpf)
        save_img(path, '4', hpf)

        # (d):
        img_pad = cv2.copyMakeBorder(img, 0, rows, 0, cols, cv2.BORDER_CONSTANT, None, value = 0)
        img_pad_f = np.fft.fft2(np.float32(img_pad))
        img_pad_fshift = np.fft.fftshift(img_pad_f)
        low_pass = img_pad_fshift * lpf
        high_pass = img_pad_fshift * hpf
        low_pass = np.fft.ifftshift(low_pass)
        high_pass = np.fft.ifftshift(high_pass)
        low_result = np.abs(np.fft.ifft2(low_pass))
        high_result = np.abs(np.fft.ifft2(high_pass))
        low_result = low_result[0:600, 0:600]
        high_result = high_result[0:600, 0:600]
        save_img(path, '5', low_result)
        save_img(path, '6', high_result)

        # (e):
        target = magnitude_spectrum[0:300, 0:300]
        top = dict()
        for col in range(300):
            for row in range(300):
                if len(top) < 25:
                    top[(col, row)] = target[col, row]
                elif target[col, row] > top[min(top, key=top.get)]:
                    key_list = list(top.keys())
                    val_list = list(top.values())
                    del top[min(top, key=top.get)]
                    top[col, row] = target[col, row]
        before = list(top.keys())
        after = []
        for col in range(cols):
            for row in range(rows):
                if (col, row) in before:
                    before.remove((col, row))
                    after.append((col, row))
        file = open(path + 'Result.txt', 'w+')
        for i in after:
            file.write('[' + str(i) + ' ' + str(target[i]) + ']' + '\n')
        