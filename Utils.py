from math import floor
import matplotlib.pyplot as plt

import numpy as np

# define sigmoid and its derivative for activation & backprop
from Settings import CATEGORIES, FOLDS, PRINCIPAL_COMPONENTS


def sigmoid(x):
    """

    :param x: input data, can be vector
    :return: sigmoid of data
    """
    return 1 / (1 + np.exp(-x))


def kfold(data):
    """
    Balanced k-fold shuffle.

    The input data is shuffled per category, but the shuffled sets of categories is added along to a single list.

    :param data: input data
    :return: folds
    """

    k = FOLDS

    fold_size = int((1 / k) * len(data))
    folds = []

    num_categories = len(CATEGORIES)
    section_size = int(len(data) / num_categories)
    for i in range(0, len(data)):
        section = i % num_categories
        section_index = floor(i / num_categories)
        fold_index = floor(i / fold_size)
        if len(folds) == fold_index:
            folds.append([])
        folds[fold_index].append(data[section * section_size + section_index])
    return folds


def show_principal_components(pca, number_components=4):
    """
    Show the principal components as images

    :param pca: pca instance
    :param number_components: number of components to show
    """
    images = pca.components
    images = images[:, :number_components]
    images = [img.reshape(122, 160) for img in images.T]
    images = np.concatenate(images, axis=1)

    plt.imshow(images, cmap='gray')
    plt.show()
