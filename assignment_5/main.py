"""
This is the main file for Assignment 5: Segmentation.
Authors: Mostafa Elshamy, Tom Vig, Andreas Hammer

"""
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import os
import otsu
from skimage.segmentation import chan_vese


def k_means_clustering(image_array,name,k=2):
    """
    K-means clustering
    1. Take grey image as input
    2. Run k-means on k=2

    """

    centroids = get_rand_centroids(image_array,k)

    image_array_label = get_image_array_label(image_array, centroids)

    print(image_array_label)
    save_image(image_array_label,name+"_k" + str(k))

    
    
def lloyds_algorithm(image_array,name,k=3):
    """
        3. Run k-means for higher k values thresholf with lloyds Algorithm
        Start with k candidate centroids. How to choose them?
        Repeat:
            1. Label pixels / assign them to clusters from their feature distance to centroids.
            2. Recompute centroids of features as average of clusters until regions or centroids don’t change.
    """
    centroids = get_rand_centroids(image_array,k)
    j=0
    while j<100:

        # Assign each pixel to the nearest centroid
        image_array_label = get_image_array_label(image_array, centroids)

        # recompute centroids based on prev run. average of pixels in each label
        new_centroids = []
        for label in range(k):
            cluster = image_array[image_array_label == label]
            new_centroid = np.mean(cluster, axis=0)
            new_centroids.append(new_centroid)

        if np.allclose(new_centroids, centroids,atol=10): #check tolerance
            save_image(image_array_label,name+"_Lloyds_k" + str(k))
            return image_array_label,centroids
        else:
            centroids = new_centroids
            j+=1
        continue
    print("Lloyds Algorithm did not converge")

def get_image_array_label(image_array, centroids): #done
    image_array_label = np.zeros(image_array.shape)
    for x in range(image_array.shape[0]):
        for y in range(image_array.shape[1]):
            image_array_label[x,y] = get_nearest_centroid(enumerate(centroids), image_array[x][y])
    return image_array_label

def assign_pixels_to_clusters(image_array, centroids): #done
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

def get_rand_centroids(image_array,k): #done
    centroids = []
    for i in range(k):
        centroids.append( [(np.random.randint(0, image_array.max()) ) , (np.random.randint(0, image_array[1].max())) ] )
    centroid_values = [image_array[centroid[0], centroid[1]] for centroid in centroids]
    return centroid_values


def save_image(image_array_label ,name):
    """
    Save the image, and color codes each segment
    Tool for clustering
    """
    
    image_array_label = image_array_label.astype(int)
    num_segments = int(image_array_label.max() + 1)
    cmap = plt.get_cmap('tab20', num_segments)  
    colors = cmap(np.arange(num_segments))[:, :3]  

    # Map each segment index to its corresponding color
    color_coded_image = colors[image_array_label]
    color_coded_image = Image.fromarray((color_coded_image * 255).astype(np.uint8))
    # Save the color-coded image

    color_coded_image.save("Results/"+name +".png")


    
def get_nearest_centroid(centroids_values, pixel_value): #done
    """
    Get the nearest centroid to a pixel
    tool for k-means clustering
    """
    min_distance = np.inf
    nearest_centroid = None
    for i, centroid in centroids_values:
        distance = float(abs(centroid - float(pixel_value)))
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
    parser = argparse.ArgumentParser(description="Run image segmentation")
    parser.add_argument("image_file", type=str, help="Path to image file for segmentation")
    parser.add_argument("-d", "--denoise", action="store_true", help="Runs denoising after segmentation")

    args = parser.parse_args()
    image_name = os.path.split(os.path.basename(args.image_file))[0]
    image = Image.open(args.image_file)
    #  Grey-scale the image
    image_grey = image.convert("L")
    image_array = np.array(image_grey)
    # k_means_clustering(image_array,image_name,2)
    # k_means_clustering(image_array,image_name,3)
    segmentation, _ = lloyds_algorithm(image_array,image_name,2)
    plt.title("LLoyds")
    plt.imshow(segmentation, cmap="gray")
    plt.show()
    # lloyds_algorithm(image_array,image_name,6)
    segmentation = otsu.otsu(image_array)
    plt.title("Otsu")
    plt.imshow(segmentation, cmap="gray")
    plt.show()
    segmentation = chan_vese(image_array)
    plt.title("Chan Vese")
    plt.imshow(segmentation, cmap="gray")
    plt.show()



if __name__ == "__main__":
    main()
