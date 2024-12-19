import cv2 as cv
import glob
import os
import numpy as np
from matplotlib import pyplot as plt


def create_codebook(training_images, k_means):
    descriptors = []
    sift = cv.SIFT_create()
    for path in training_images:
        img = cv.imread(path)
        gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        _, des = sift.detectAndCompute(gray, None)
        descriptors.append(des)        
    matrix = np.vstack(descriptors)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv.KMEANS_RANDOM_CENTERS
    compactness, labels, codebook = cv.kmeans(matrix, k_means, None, criteria, 10, flags)
    return compactness, labels, codebook

def main():
    dataset_folder = "101_ObjectCategories"
    images_per_category = 20
    selected_categories = ["brain","cannon","ant","octopus","butterfly"]
    training_images = []
    test_images = []
    for category in os.listdir(dataset_folder):
        if category in selected_categories:
            category_path = os.path.join(dataset_folder, category, '*.jpg')
            image_files = glob.glob(category_path)
            for image in image_files[:images_per_category // 2]:
                training_images.append(image)
            for image in image_files[images_per_category // 2:images_per_category]:
                test_images.append(image)

    k_means_values = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800]
    compactness_values = []

    for k in k_means_values:
        print("---Running test for k:{}---".format(k))
        compactness, _, _ = create_codebook(training_images, k)
        compactness_values.append(compactness)

    plt.plot(k_means_values, compactness_values)
    plt.show()

if __name__ == "__main__":
    main()
