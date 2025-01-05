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


def run_elbow_test(training_images):
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
    query_hist = np.array(query_hist)

    # Calculate 'common words' similarity
    def common_words_similarity(hist):
        return np.sum(np.minimum(query_hist, np.array(hist)))

    # Create a copy of the dataset to avoid SettingWithCopyWarning
    dataset_copy = dataset.copy()

    # Apply similarity calculation to each row in the dataset
    dataset_copy.loc[:, 'similarity'] = dataset_copy['hist'].apply(common_words_similarity)

    # Sort by similarity in descending order
    ranked_dataset = dataset_copy.sort_values(by='similarity', ascending=False)

    return ranked_dataset


def experiment_1(training_data):
    reciprocal_ranks = []
    correct_in_top3 = 0
    total_queries = len(training_data)

    for idx, query_row in training_data.iterrows():
        query_hist = query_row['hist']
        true_label = query_row['true_label']

        # Perform retrieval
        ranked_dataset = retreive(training_data, query_hist)

        # Calculate rank of the first correct match
        rank = 1
        for _, row in ranked_dataset.iterrows():
            if row['true_label'] == true_label:
                reciprocal_ranks.append(1 / rank)
                if rank <= 3:
                    correct_in_top3 += 1
                break
            rank += 1

    # Calculate MRR
    mrr = sum(reciprocal_ranks) / total_queries

    # Calculate Top-3 Accuracy
    top3_accuracy = (correct_in_top3 / total_queries) * 100

    # Print final metrics
    print("\nExperiment 1 Summary:")
    print(f"Total Queries: {total_queries}")
    print(f"Mean Reciprocal Rank (MRR): {mrr:.4f}")
    print(f"Top-3 Accuracy: {top3_accuracy:.2f}%")

    return mrr, top3_accuracy


def main():
    data_file_name = 'data'
    dataset_folder = "101_ObjectCategories"
    images_per_category = 20
    selected_categories = ["brain", "cannon", "ant", "octopus", "butterfly"]
    training_images, test_images = read_dataset(dataset_folder, images_per_category, selected_categories)

    # run_elbow_test(training_images)

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
    # query_train_image = data[data['is_train'] == True].iloc[0]
    # print(data.dtypes)
    # print("Running Retreival on {}".format(query_train_image))
    # ranking = retreive(data, query_train_image['hist'])
    # print(ranking.head(5))

    experiment_1(data[data['is_train'] == True])


if __name__ == "__main__":
    main()
