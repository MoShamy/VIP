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


def run_elbow_test(training_images, num_categories):
    k_means_values = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600]
    compactness_values = []
    for k in k_means_values:
        print("---Running test for k:{}---".format(k))
        compactness, _, _ = create_codebook(training_images, k)
        compactness_values.append(compactness)

    plt.plot(k_means_values, compactness_values)
    plt.savefig("results/elbow_test_plot_{}_cat.png".format(num_categories))
    plt.close()


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


def experiment(training_data, testing_data):
    results = {}
    overall_reciprocal_ranks = []
    overall_correct_in_top3 = 0
    total_queries = len(testing_data)

    categories = testing_data['true_label'].unique()

    for category in categories:
        category_data = testing_data[testing_data['true_label'] == category]
        category_reciprocal_ranks = []
        category_correct_in_top3 = 0
        total_category_queries = len(category_data)

        for idx, query_row in category_data.iterrows():
            query_hist = query_row['hist']
            true_label = query_row['true_label']

            ranked_dataset = retreive(training_data, query_hist)
            ranked_dataset = ranked_dataset[ranked_dataset['path'] != query_row['path']]

            rank = 1
            for _, row in ranked_dataset.iterrows():
                if row['true_label'] == true_label:
                    category_reciprocal_ranks.append(1 / rank)
                    overall_reciprocal_ranks.append(1 / rank)
                    if rank <= 3:
                        category_correct_in_top3 += 1
                        overall_correct_in_top3 += 1
                    break
                rank += 1

        mrr = sum(category_reciprocal_ranks) / total_category_queries if total_category_queries > 0 else 0
        top3_accuracy = (category_correct_in_top3 / total_category_queries) * 100 if total_category_queries > 0 else 0

        results[category] = {'MRR': mrr, 'Top3_Accuracy': top3_accuracy}

    overall_mrr = sum(overall_reciprocal_ranks) / total_queries if total_queries > 0 else 0
    overall_top3_accuracy = (overall_correct_in_top3 / total_queries) * 100 if total_queries > 0 else 0

    results['Overall'] = {'MRR': overall_mrr, 'Top3_Accuracy': overall_top3_accuracy}

    return results


def main():
    ex_num = 2 # 1 or 2
    images_per_category = 20
    num_categories = 20 # 5 or 20
    k = 300
    load_train_data = True

    dataset_folder = "101_ObjectCategories"
    data_file_name = 'data_{}'.format(num_categories)
    selected_categories = []
    if num_categories == 5:
        selected_categories = ["brain", "cannon", "ant", "octopus", "butterfly"]
    elif num_categories == 20:
        selected_categories = ["brain", "cannon", "ant", "octopus", "butterfly", "buddha", "butterfly", "camera",
                               "cellphone", "chair", "crocodile", "dolphin", "elephant", "emu", "flamingo", "headphone",
                               "lamp", "lotus", "menorah", "pigeon"]
    training_images, test_images = read_dataset(dataset_folder, images_per_category, selected_categories)

    # run_elbow_test(training_images)


    # Load from file if we have already run the indexing before
    if load_train_data:
        data = pd.read_csv('{}.csv'.format(data_file_name))

        # Convert 'hist' from string to Python list, then to NumPy array
        data['hist'] = data['hist'].apply(ast.literal_eval)  # Parse string to list
        data['hist'] = data['hist'].apply(np.array)  # Convert list to NumPy array
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
        data['hist'] = data['hist'].apply(lambda x: x.tolist())  # Convert NumPy arrays to lists
        data.to_csv('{}.csv'.format(data_file_name), index=False)

    training_data = data[data['is_train'] == True]
    testing_data = data[data['is_train'] == False]

    if ex_num == 1:
        results = experiment(training_data, training_data)
    elif ex_num == 2:
        results = experiment(training_data, testing_data)

    categories = [category for category in results.keys() if category != 'Overall']
    mrr_values = [metrics['MRR'] for category, metrics in results.items() if category != 'Overall']
    top3_values = [metrics['Top3_Accuracy'] for category, metrics in results.items() if category != 'Overall']

    plt.figure(figsize=(10, 5))
    plt.bar(categories, mrr_values, color='blue', alpha=0.7)
    plt.title('Mean Reciprocal Rank (MRR) per Category')
    plt.xlabel('Categories')
    plt.ylabel('MRR')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("results/mrr_{}_cat_ex{}.png".format(num_categories, ex_num))
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(categories, top3_values, color='green', alpha=0.7)
    plt.title('Top-3 Accuracy per Category')
    plt.xlabel('Categories')
    plt.ylabel('Top-3 Accuracy (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("results/top3_acc_{}_cat_ex{}.png".format(num_categories, ex_num))
    plt.close()

    # After computing the results, print the per-category results
    for category, metrics in results.items():
        if category != 'Overall':
            print(f"Category: {category}, MRR: {metrics['MRR']:.4f}, Top-3 Accuracy: {metrics['Top3_Accuracy']:.2f}%")

    # Print the overall results
    print("\nOverall Results:")
    print(f"Mean Reciprocal Rank (MRR): {results['Overall']['MRR']:.4f}")
    print(f"Top-3 Accuracy: {results['Overall']['Top3_Accuracy']:.2f}%")


    # Retrieve the first "chair" image
    query_image = testing_data[(testing_data['true_label'] == "cannon")].iloc[7]

    # Print the query image path and true label
    print(f"Query Image Path: {query_image['path']}")
    print(f"Query Image True Label: {query_image['true_label']}")

    # Perform retrieval
    ranking = retreive(training_data, query_image['hist'])

    # Exclude the query image itself from the results
    ranking = ranking[ranking['path'] != query_image['path']]

    # Get the top 3 results
    top_3_results = ranking.head(3)

    # Print the top 3 results
    print("Top 3 Results:")
    print(top_3_results[['path', 'true_label', 'similarity']])



if __name__ == "__main__":
    main()
