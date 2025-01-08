"""
This is the main file for Assignment 5: Segmentation.
Authors: Mostafa Elshamy, Tom Vig, Andreas Hammer

"""
import numpy as np
from PIL import Image

def k_means_clustering(image,name):
    """
    K-means clustering
    1. Take grey image as input
    2. Run k-means on k=2
    3. Run k-means for higher k values thresholf with lloyds Algorithm
        Start with k candidate centroids. How to choose them?
        Repeat:
            1. Label pixels / assign them to clusters from their feature distance to centroids.
            2. Recompute centroids of features as average of clusters until regions or centroids don’t change.
    """
    
    # 1.
    image_array = np.array(image)
    
    
    # 2. Run k-means on k=2
    k = 2
    # get centroids
    centroids = get_rand_centroids(image_array,k)

    #repeat until centroids don't change
    while True:

        # Assign each pixel to the nearest centroid
        image_array_label = get_image_array_label(image_array, centroids)


        # 3. Run k-means for higher k values threshold with lloyds Algorithm
        # recompute centroids based on prev run. average of pixels in each label

        for label in range(k):
            cluster = image_array[image_array_label == label]
            centroids[label] = np.mean(cluster, axis=0)
        # Assign each pixel to the nearest centroid
        image_array_label = assign_pixels_to_clusters(image_array, centroids)

        # recompute centroids based on prev run. average of pixels in each label
        for label in range(k):
            cluster = image_array[image_array_label == label]
            new_centroid = int(np.mean(cluster, axis=0))
            
            if np.array_equal(new_centroid, centroids[label]):
                save_image(image_array,image_array_label,name)
                break
            else:
                centroids[label] = new_centroid
            continue
    

def get_image_array_label(image_array, centroids):
    image_array_label = np.zeros(image_array.shape)
    for x in range(image_array.shape[0]):
        for y in range(image_array.shape[1]):
            image_array_label[x,y] = get_nearest_centroid(enumerate(centroids), image_array[x][y])
    return image_array_label

def assign_pixels_to_clusters(image_array, centroids):
    """
    Assign each pixel to the nearest centroid
    tool for k-means clustering
    """
    image_array_label = np.zeros(image_array.shape)
    for x in range(image_array.shape[0]):
        for y in range(image_array.shape[1]):
            nearest_centroid = get_nearest_centroid(image_array,enumerate(centroids), image_array[x][y])
            image_array_label[x,y] = nearest_centroid
    return image_array_label

def get_rand_centroids(image_array,k):
    centroids = []
    for i in range(k):
        centroids.append( [(np.random.randint(0, image_array.max()) ) , (np.random.randint(0, image_array[1].max())) ] )
    centroid_values = [image_array[centroid[0], centroid[1]] for centroid in centroids]
    return centroid_values


def save_image(image_array_label ,name):
    """
    Save the image, and increase contrast between labels
    Tool for clustering
    """
    # Increase contrast between labels
    # unique_labels = np.unique(image_array_label)
    # label_map = {label: i * (255 // (len(unique_labels) - 1)) for i, label in enumerate(unique_labels)}
    # high_contrast_image_array = np.vectorize(label_map.get)(image_array_label)

    # Convert to image, showing the labels in different colors
    image_label = Image.fromarray(image_array_label.astype(np.uint8))

    image_label.save("Results/"+name +"_2k.png")


    
def get_nearest_centroid(image_array,centroids, pixel):
    """
    Get the nearest centroid to a pixel
    tool for k-means clustering
    """
    min_distance = np.inf
    nearest_centroid = None
    for i, centroid in centroids:
        distance = np.linalg.norm(np.array(image_array[centroid]) - np.array(image_array[pixel]))
        if distance < min_distance:
            min_distance = distance
            nearest_centroid = i
    return nearest_centroid 



def main():
    """
    1. K-means clustering
    2. Otsu's thresholding
    3. cleaning/denoising
    """
    image_name = "coins"
    image = Image.open("images/"+ image_name +".png")
    #  Grey-scale the image
    image_grey = image.convert("L")
    k_means_clustering(image_grey,image_name)



if __name__ == "__main__":
    main()