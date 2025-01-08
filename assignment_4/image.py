import numpy as np
import cv2 as cv

class Image:
    def __init__(self, path, category):
        self.path = path
        self.data = cv.cvtColor(cv.imread(path), cv.COLOR_BGR2GRAY)
        self.descriptors = None
        self.category = category

    def get_category(self):
        return self.category

    def get_path(self):
        return self.path

    def get_data(self):
        return self.data

    def set_descriptors(self, descriptors):
        self.descriptors = descriptors

    def get_descriptors(self):
        return self.descriptors

    def get_histogram(self, centers, k):
        distances = np.linalg.norm(self.descriptors[:, np.newaxis] - centers, axis=2)

        closest_centers = np.argmin(distances, axis=1)

        counts = np.bincount(closest_centers, minlength=k)

        normalized_hist = counts / len(closest_centers)

        return np.array(normalized_hist) 

