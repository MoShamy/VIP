import numpy as np
from PIL import Image


def get_neighbour_labels(image_array_label, x, y, neighbour_system):
    label_count = {}
    if neighbour_system == 4:
        indexes = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
    else:  # num_neighbours == 8
        indexes = [(x + i, y + j) for i in range(-1, 2) for j in range(-1, 2) if i != 0 or j != 0]

    for i, j in indexes:
        if 0 <= i < image_array_label.shape[0] and 0 <= j < image_array_label.shape[1]:
            label = image_array_label[i, j]
            if label in label_count:
                label_count[label] += 1
            else:
                label_count[label] = 1
    return label_count


def run_clean(image_array_label, neighbour_system, threshold, iterations):
    assert image_array_label.ndim == 2

    for i in range(iterations):
        for x in range(image_array_label.shape[0]):
            for y in range(image_array_label.shape[1]):
                # get the label values of the neighbours
                label_count = get_neighbour_labels(image_array_label, x, y, neighbour_system)

                # get the most common label and number of times it appears
                max_label = max(label_count, key=label_count.get)
                max_count = label_count[max_label]

                # may vary for corner cases and the like
                num_neighbours = sum(label_count.values())

                if max_count >= threshold * num_neighbours:
                    image_array_label[x, y] = max_label
    return image_array_label


