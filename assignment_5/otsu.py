import numpy as np


# Return binary image based on optimal threshold value
def otsu(image):
    
    min = np.min(image)
    max = np.max(image)

    t_range = np.arange(min, max)

    optimal = [0, np.inf]

    for t in t_range:
        variance = intra_class_variance(image, t)
        if variance < optimal[1]:
            optimal[0] = t
            optimal[1] = variance

    return (image > optimal[0]).astype(np.uint8)

def intra_class_variance(image, t):
    histogram = image.flatten()

    class1 = histogram[histogram > t]
    class2 = histogram[histogram <= t]

    w1 = len(class1) / len(histogram)
    w2 = len(class2) / len(histogram)

    sigma1 = np.var(class1)
    sigma2 = np.var(class2)

    return w1*sigma1 + w2*sigma2
