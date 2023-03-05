# NYCU_DIP
2022 NYCU 111-1 Image Processing

## Homework 1
Plot all the intermediate (in-process) images:
(a) original, (b) Laplacian, (c) Laplacian-sharpened, (d) Sobel-gradient, (e) smoothed gradient. (f) extracted feature: (e)x(b), (g) (a)+(f), and (h) final image obtained by power-law transformation of (g).

## Homework 2
Apply Gaussian lowpass and highpass filters to two images, kid and fruit image.
Assume cutoff frequency D0 = 100 pixels (based on the original image size) for lowpass and highpass filters.

## Homework 3
For the degraded images Kid2 degraded.tif, try to restore the original image by
Step 1: De-noise (reduce noise power), and Step 2: Inverse filtering (deconvolution) method provided that the image was blurred by Gaussian model

## Homework 4
Consider the RGB color image love&peace rose.tif.
1. Determine and plot the R, G and B component images, as well as the H, S and I component images.
2. Enhance the image by both RGB-sharpening and HSI-sharpening scheme.

## Homework 5
Consider the gray-scale image, Kid at playground.tif, apply Canny edge detection algorithm to obtain the edge image by using the following setup and parameters:
• sigma of Gaussian smoothing filter: 0.5% of the shortest dimension of the image
• Sobel operator for computing gradient vectors
• Hysteresis thresholding: TH = 0.10 TL = 0.04
