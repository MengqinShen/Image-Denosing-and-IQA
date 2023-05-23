# Used to view the images
import matplotlib.pyplot as plt
# Used to perform filtering on an image
import cv2
# Used to create kernels for filtering
import numpy as np
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
from skimage import data, img_as_float

# plot teo images side by side
def plot_image(image_1, image_2,title_1="Orignal",title_2="New Image"):
    plt.figure(figsize=(10,10))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB))
    plt.title(title_1)
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(image_2, cv2.COLOR_BGR2RGB))
    plt.title(title_2)
    plt.show()

# plot four images side by side
def plot_image4(image_1, image_2,image_3,image_4,title_1="Orignal",title_2="New Image1",title_3="New Image2",title_4="New Image3"):
    plt.figure(figsize=(10,10))
    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB))
    plt.title(title_1)
    plt.subplot(2, 2, 2)
    plt.imshow(cv2.cvtColor(image_2, cv2.COLOR_BGR2RGB))
    plt.title(title_2)
    plt.subplot(2, 2, 3)
    plt.imshow(cv2.cvtColor(image_3, cv2.COLOR_BGR2RGB))
    plt.title(title_3)
    plt.subplot(2, 2, 4)
    plt.imshow(cv2.cvtColor(image_4, cv2.COLOR_BGR2RGB))
    plt.title(title_4)
    plt.show()


image = cv2.imread("lenna.jpg")
# image = img_as_float("lenna.jpg")
# plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# Get the number of rows and columns in the image
rows, cols,_ = np.shape(image)
# Creates values using a normal distribution with a mean of 0 and standard deviation of 15, the values are converted to unit8 which means the values are between 0 and 255
noise = np.random.normal(0,15,(rows,cols,3)).astype(np.uint8)
# Add the noise to the image
noisy_image = image + noise
# Plots the original image and the image with noise using the function defined at the top
# plot_image(image, noisy_image, title_1="Orignal",title_2="Image + Noise")

# filtering noise
## 1. box filter
# Create two kernels, one 6 by 6 array where each value is 1/36, another4 by 4 array where each value is 1/16(keeps the image sharp, but filters less noise)
kernel1 = np.ones((6,6))/36
kernel2 = np.ones((4,4))/16
# Filters the images using the kernel
image_filtered1 = cv2.filter2D(src=noisy_image, ddepth=-1, kernel=kernel1)
image_filtered2 = cv2.filter2D(src=noisy_image,ddepth=-1,kernel=kernel2)
# plot_image4(image,noisy_image,image_filtered1, image_filtered2,title_1="Original image",\
#             title_2="Image Plus Noise",title_3="Box Filtered image1",title_4=" Box Filtered image2")
## 2. GaussianBlur
# Filters the images using GaussianBlur on the image with noise using a 4 by 4 kernel and a 11 by 11 kernel
image_filtered3 = cv2.GaussianBlur(noisy_image,(5,5),sigmaX=4,sigmaY=4)
image_filtered4 = cv2.GaussianBlur(noisy_image,(11,11),sigmaX=10,sigmaY=10)
# Plots the Filtered Image then the Unfiltered Image with Noise
# plot_image4(image,noisy_image,image_filtered3, image_filtered4,title_1="Original image",title_2="Image Plus Noise",\
#             title_3="Filtered image GaussianBlur1",title_4="Filtered image GaussianBlur2")

## Image Sharpening
# Common Kernel for image sharpening
kernel = np.array([[-1,-1,-1],
                   [-1, 9,-1],
                   [-1,-1,-1]])
# Applies the sharpening filter using kernel on the original image without noise
sharpened = cv2.filter2D(image, -1, kernel)
# Plots the sharpened image and the original image without noise
# plot_image( image,sharpened ,title_1="Original image", title_2="Sharpened image")

# comparing filtered noised images to original images for MSE, PSNR, SSIM, CNR, and MTF
# Convert to grayscale
imgOrig_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
imgNoise_gray = cv2.cvtColor(noisy_image, cv2.COLOR_BGR2GRAY)
imgBox_gray = cv2.cvtColor(image_filtered1, cv2.COLOR_BGR2GRAY)
imgGussian_gray = cv2.cvtColor(image_filtered3, cv2.COLOR_BGR2GRAY)

# Calculate MSE
mse = mean_squared_error(imgOrig_gray, imgOrig_gray)
mse1 = mean_squared_error(imgOrig_gray, imgNoise_gray)
mse2 = mean_squared_error(imgOrig_gray, imgBox_gray)
mse3 = mean_squared_error(imgOrig_gray, imgGussian_gray)
print(f'MSE: {mse:.4f} ,{mse1:.4f},{mse2:.4f},{mse:.4f}dB')
# Calculate PSNR
psnr = 0
psnr1 = peak_signal_noise_ratio(imgOrig_gray, imgNoise_gray)
psnr2 = peak_signal_noise_ratio(imgOrig_gray, imgBox_gray)
psnr3 = peak_signal_noise_ratio(imgOrig_gray, imgGussian_gray)
print(f'PSNR: {psnr:.4f},{psnr1:.4f},{psnr2:.4f},{psnr3:.4f} ')
# Calculate SSIM
ssim = structural_similarity(imgOrig_gray, imgOrig_gray)
ssim1 = structural_similarity(imgOrig_gray, imgNoise_gray)
ssim2 = structural_similarity(imgOrig_gray, imgBox_gray)
ssim3 = structural_similarity(imgOrig_gray, imgGussian_gray)
print(f'SSIM: {ssim:.4f},{ssim1:.4f},{ssim2:.4f},{ssim3:.4f} ')
# Calculate CNR
# Assuming the images have a uniform background with mean m and standard deviation s
m, s = cv2.meanStdDev(imgOrig_gray)
m1, s1 = cv2.meanStdDev(imgNoise_gray)
m2, s2 = cv2.meanStdDev(imgBox_gray)
m3, s3 = cv2.meanStdDev(imgGussian_gray)

cnr = m / s # CNR of image 1
cnr_float = float(cnr)
cnr1 = m1 / s1 # CNR of image +noise
cnr1_float = float(cnr1)
cnr2 = m2 / s2 # CNR of image +noise+box filter
cnr2_float = float(cnr2)
cnr3 = m3 / s3 # CNR of image +noise +gaussian filter
cnr3_float = float(cnr3)
# print('CNR ='+str(cnr),'CNR1='+str(cnr1),'CNR2='+str(cnr2),'CNR3='+str(cnr3))
# print(type(cnr), type(mse))
print(f'CNR: {cnr_float:.2f},{cnr1_float:.2f},{cnr2_float:.2f},{cnr3_float:.2f} ')
# Calculate MTF
# Assuming the images have a slanted edge with angle theta and spatial frequency f
theta = np.pi / 4 # 45 degrees
f = 10 # cycles per pixel
mtf = np.abs(np.fft.fft2(imgOrig_gray * np.sin(2 * np.pi * f * np.cos(theta)))) # MTF of image original
mtf1 = np.abs(np.fft.fft2(imgOrig_gray * np.sin(2 * np.pi * f * np.cos(theta)))) # MTF of image 1
mtf2 = np.abs(np.fft.fft2(imgBox_gray * np.sin(2 * np.pi * f * np.cos(theta)))) # MTF of image 2
mtf3 = np.abs(np.fft.fft2(imgGussian_gray * np.sin(2 * np.pi * f * np.cos(theta)))) # MTF of image 3


#
# plot_image4(image,noisy_image,image_filtered1, image_filtered3,title_1="Original image",title_2="Image Plus Noise",\
#             title_3="Filtered image BoxBlur",title_4="Filtered image GaussianBlur")

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8),
                         sharex=True, sharey=True)
ax = axes.ravel()
ax[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# ax[0].set_xlabel(f'MSE: {mse:.2f},MTF: {mtf:.2f}')
ax[0].set_xlabel(f'MSE: {mse:.2f},PSNR: {psnr:.2f},SSIM: {ssim:.2f},CNR: {cnr_float:.2f}')
ax[0].set_title('Original image')
ax[1].imshow(cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB))
ax[1].set_xlabel(f'MSE: {mse1:.2f},PSNR: {psnr1:.2f},SSIM: {ssim1:.2f},CNR: {cnr1_float:.2f}')
ax[1].set_title('Image with noise')

ax[2].imshow(cv2.cvtColor(image_filtered1, cv2.COLOR_BGR2RGB))
ax[2].set_xlabel(f'MSE: {mse2:.2f},PSNR: {psnr2:.2f},SSIM: {ssim2:.2f},CNR: {cnr2_float:.2f}')
ax[2].set_title('Noised Image plus box filter')

ax[3].imshow(cv2.cvtColor(image_filtered3, cv2.COLOR_BGR2RGB))
ax[3].set_xlabel(f'MSE: {mse3:.2f},PSNR: {psnr3:.2f},SSIM: {ssim3:.2f},CNR: {cnr3_float:.2f}')
ax[3].set_title('Noised Image plus Gaussian filter')
plt.tight_layout()
plt.show()



## edges Sobel filter

img_gray = cv2.imread('lenna.jpg', cv2.IMREAD_GRAYSCALE)
# Filters the images using GaussianBlur on the image with noise using a 3 by 3 kernel to smooth the image,\
# which decreases changes that may be caused by noise that would affect the gradient
img_gray = cv2.GaussianBlur(img_gray,(3,3),sigmaX=0.1,sigmaY=0.1)
ddepth = cv2.CV_16S
# Applys the filter on the image in the X direction, ksize must be 1, 3, 5, or 7
grad_x = cv2.Sobel(src=img_gray, ddepth=ddepth, dx=1, dy=0, ksize=3)
grad_y = cv2.Sobel(src=img_gray, ddepth=ddepth, dx=0, dy=1, ksize=3)
# Converts the values back to a number between 0 and 255
abs_grad_x = cv2.convertScaleAbs(grad_x)
abs_grad_y = cv2.convertScaleAbs(grad_y)

# Adds the derivative in the X and Y direction
grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
plt.figure(figsize=(10,10))
plt.subplot(1, 3, 1)
plt.imshow(abs_grad_x,cmap='gray')
plt.title("grad x")
plt.subplot(1, 3, 2)
plt.imshow(abs_grad_y,cmap='gray')
plt.title("grad y")
plt.subplot(1, 3, 3)
plt.imshow(grad,cmap='gray')
plt.title("grad weighted")
plt.show()

#median filters to  see how a median filter improves segmentation
image = cv2.imread("cameraman.jpeg",cv2.IMREAD_GRAYSCALE)
# Filter the image using Median Blur with a kernel of size 5
filtered_image = cv2.medianBlur(image, 5)
# Make the image larger when it renders
plt.figure(figsize=(10,10))
plt.subplot(1, 2, 1)
plt.imshow(image,cmap='gray')
plt.title("Original image")
plt.subplot(1, 2, 2)
plt.imshow(filtered_image,cmap='gray')
plt.title("medianBlurd image")
plt.show()
# Returns ret which is the threshold used and outs which is the image
ret, outs = cv2.threshold(src = image, thresh = 0, maxval = 255, type = cv2.THRESH_OTSU+cv2.THRESH_BINARY_INV)
# Make the image larger when it renders
plt.figure(figsize=(10,10))

# Render the image
plt.imshow(outs, cmap='gray')
plt.show()