import cv2 as cv
import glob
import os
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from scipy.stats import entropy
import ast

from image import Image


def create_codebook(training_images, k_means):
    descriptors = []
    sift = cv.SIFT_create()
    for im in training_images:
        data = im.get_data()
        _, des = sift.detectAndCompute(data, None)
        descriptors.append(des)        
        im.set_descriptors(des)
    matrix = np.vstack(descriptors)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv.KMEANS_RANDOM_CENTERS
    compactness, labels, codebook = cv.kmeans(matrix, k_means, None, criteria, 10, flags)
    return compactness, labels, codebook

def read_dataset(dataset_folder, images_per_category, selected_categories):
    training_images = []
    test_images = []
    for category in os.listdir(dataset_folder):
        if category in selected_categories:
            category_path = os.path.join(dataset_folder, category, '*.jpg')
            image_files = glob.glob(category_path)
            for image in image_files[:images_per_category // 2]:
                training_images.append(Image(image, category))
            for image in image_files[images_per_category // 2:images_per_category]:
                test_images.append(Image(image, category))
    return training_images, test_images

def run_elbow_test():
    k_means_values = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600]
    compactness_values = []
    for k in k_means_values:
        print("---Running test for k:{}---".format(k))
        compactness, _, _ = create_codebook(training_images, k)
        compactness_values.append(compactness)

    plt.plot(k_means_values, compactness_values)
    plt.show()

# The retreival function
#
# Data - is the pandas dataframe either loaded from file or created during the indexing process
#
# query_histogram - the query image histogram. The 'hist' column from the dataframe, which can either be
# from train or test set depending on experiment.
#
# returns a pandas dataframe with same columns as 'Data' but with an additional column 'similarity' which
# is the similarity score based on 'common words' which the dataframe is sorted by.

def retreive(dataset, query_hist):
    # Ensure query_hist is a NumPy array
    histograms = dataset['hist'].apply(np.array).tolist()

    # Step 2: Normalize histograms (ensure they sum to 1)
    histograms = [hist / hist.sum() for hist in histograms]
    query_hist /= query_hist.sum()

    # Step 3: Compute Kullback-Leibler Divergence between the query and each dataset histogram
    kl_divergences = []
    for hist in histograms:
        kl_div = entropy(query_hist, hist)  # KL Divergence from query_hist to hist
        kl_divergences.append(kl_div)

    # Step 4: Add KL divergence values to the dataset and sort by divergence
    dataset['kl_divergence'] = kl_divergences
    ranked_dataset = dataset.sort_values(by='kl_divergence', ascending=True)
    
    return ranked_dataset

def main():
    data_file_name = 'data'
    dataset_folder = "101_ObjectCategories"
    images_per_category = 20
    selected_categories = ["brain","cannon","ant","octopus","butterfly"]
    training_images, test_images = read_dataset(dataset_folder, images_per_category, selected_categories)

    #run_elbow_test()

    k = 300
    # Boolean to use file or generate new file, Change to false when updating any parameters above.
    load_train_data = False

    # Load from file if we have already run the indexing before
    if load_train_data:
        data = pd.read_csv('{}.csv'.format(data_file_name))
    else:
        columns = ['path', 'true_label', 'is_train', 'hist']
        data = pd.DataFrame(columns=columns)

        # Create codebook
        _, _, codebook = create_codebook(training_images, k_means=k)

        # Get test descriptors
        _, _, _ = create_codebook(test_images, k_means=k)

        # Create histograms and populate dataframe
        for im in training_images:
            data.loc[len(data)] = [im.get_path(), im.get_category(), True, im.get_histogram(codebook, k)]
        for im in test_images:
            data.loc[len(data)] = [im.get_path(), im.get_category(), False, im.get_histogram(codebook, k)]
            
        # Save indexing to file
        data.to_csv('{}.csv'.format(data_file_name), index=False)

    # Get some trainig image.
    query_train_image = data[data['is_train'] == True].iloc[0]
    print(data.dtypes)
    print("Running Retreival on {}".format(query_train_image))
    ranking = retreive(data, query_train_image['hist'])
    print(ranking.head(5))


    
        
if __name__ == "__main__":
    main()
