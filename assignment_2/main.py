import os
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_laplace, gaussian_gradient_magnitude
from skimage.io import imread, imsave
from skimage.feature import canny
import matplotlib.pyplot as plt

image_folder = "images"

def filter_plots(im, sigma_vals, filter):
    """
    Plot the image after applying the filter for different sigma values
    """
    print("Plotting for {}".format(filter.__name__))
    n = len(sigma_vals)
    for i in range(n):
        sigma = sigma_vals[i]
        plt.subplot(2, n//2, i+1)
        plt.title("Filter = {}, Sigma = {}".format(filter.__name__, sigma))
        plt.imshow(filter(input=im, sigma=sigma), cmap='gray')
    plt.show()

def canny_plots(im, low_vals, high_vals, sigma_vals):
    """
    Plot the image after applying Canny edge detection for different low and high threshold values
    """
    print("Plotting Canny edge detection")
    n = len(low_vals)
    for i in range(n):
        low = low_vals[i]; high = high_vals[i]; sigma = sigma_vals[i];
        plt.subplot(2, n // 2, i+1)
        plt.title("Low = {}, High = {}, Sigma = {}".format(low, high, sigma))
        plt.imshow(canny(im, low_threshold=low, high_threshold=high, sigma=sigma), cmap='gray')
    plt.show()

def make_square(image_size, square_size):
    """
    Make image square
    """
    image = np.ones(image_size)
    start_row = (image_size[0] - square_size) // 2
    start_col = (image_size[1] - square_size) // 2
    image[start_row:start_row+square_size, start_col:start_col+square_size] = 0
    return image

def main():
    # Load Image
    image_name = "peppers.jpg" #Change the image name to test different images ie. "peppers.jpg" or 

    image = imread("assignment2/images/" + image_name, as_gray=True) #Set as gray to make the image grayscale
    

    
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
