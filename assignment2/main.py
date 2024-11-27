import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_laplace, gaussian_gradient_magnitude
from skimage.io import imread, imsave
from skimage.feature import canny
import matplotlib.pyplot as plt

image_folder = "images/"

def filter_plots(im, sigma_vals, filter):
    print("Plotting for {}".format(filter.__name__))
    n = len(sigma_vals)
    for i in range(n):
        sigma = sigma_vals[i]
        plt.subplot(2, n//2, i+1)
        plt.title("Filter = {}, Sigma = {}".format(filter.__name__, sigma))
        plt.imshow(filter(input=im, sigma=sigma), cmap='gray')
    plt.show()

def canny_plots(im, low_vals, high_vals, sigma_vals):
    print("Plotting Canny edge detection")
    n = len(low_vals)
    for i in range(n):
        low = low_vals[i]; high = high_vals[i]; sigma = sigma_vals[i];
        plt.subplot(2, n // 2, i+1)
        plt.title("Low = {}, High = {}, Sigma = {}".format(low, high, sigma))
        plt.imshow(canny(im, low_threshold=low, high_threshold=high, sigma=sigma), cmap='gray')
    plt.show()

def main():
    # Load Image
    image_name = "mandrill.jpg"
    image = imread(image_folder+image_name, as_gray=True)

    # Define Sigma Values for experiment
    sigma_vals = [1,2,4,8]

    filter_plots(image, sigma_vals, gaussian_filter)

    filter_plots(image, sigma_vals, gaussian_gradient_magnitude)

    filter_plots(image, sigma_vals, gaussian_laplace)

    # Define Low and High Threshold Values and Sigma Values for Canny Edges Detection
    low_vals = [0,0,0,0.1,0.2,0.2]
    high_vals = [0.1, 0.2, 0.4, 0.4, 0.4, 0.4]
    sigma_vals = [1, 1, 1, 1, 1, 2]
    canny_plots(image, low_vals=low_vals, high_vals=high_vals, sigma_vals=sigma_vals) 

if __name__=="__main__":
    main()
