import matplotlib.pyplot as plt
import numpy as np

from LogisticRegression import LogisticRegression
from PCA import PCA
from Settings import LOGISTIC, CATEGORIES, LEARNING_RATE, PRINCIPAL_COMPONENTS, EPOCHS, STOCHASTIC_GRADIENT
from SoftmaxRegression import SoftmaxRegression
from Utils import kfold


class EpochData:
    def __init__(self):
        self.acc = []
        self.error = []

    def save(self, error, acc):
        """
        Record data for a single epoch
        :param error: cross entropy loss
        :param acc: accuracy
        """
        self.error.append(error)
        self.acc.append(acc)

    def add(self, epoch_data):
        """
        Concat EpochData objects

        :param epoch_data: EpochData object
        """
        self.error.append(epoch_data.error)

    def align(self, k):
        """
        Align a collection of EpochData

        :param k: number of collections
        """
        self.error = np.array(self.error).reshape((k, EPOCHS))

    def score(self):
        """
        Get the last loss score

        :return: score
        """
        return self.error[-1]


def transform(pca, data):
    """
    Transform data from dataloader to PCA and one-hot encoded data
    :param pca: pca instance
    :param data: input data
    :return:
    """
    labels = data[1]
    if not LOGISTIC:  # if softmax
        labels = one_hot_encode(labels)
    return pca.transform(data[0]), labels


def visualize_data(plots, legends, x_label, y_label):
    """
    Create plots

    :param plots: data to plot
    :param legends: legends on plot
    :param x_label: x label
    :param y_label: y label
    """
    x = np.arange(1, len(plots[0]) + 1)
    for data in plots:
        plt.plot(x, data)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.legend(legends)
    plt.show()


def visualize_data_avg(train_data, val_data):
    """
    Create a plot over a series with epochs data. Includes standard deviation

    :param train_data: train data
    :param val_data: validation data
    """
    x = np.arange(1, len(train_data.error[0]) + 1)
    for data in [train_data.error, val_data.error]:
        mean = np.sum(data, axis=0) / data.shape[0]  # divide by folds
        x_std = []
        y_std = []

        for i in range(0, 50):
            if (i + 1) % 10 == 0 or i == 0:
                y_std.append(np.std(data[:, i])/2)
            else:
                y_std.append(0)
        plt.errorbar(np.arange(1, 50 + 1, 1), mean, y_std)
    plt.show()


def split_x_y(data):
    """
    Unzip data

    :param data: zipped data
    :return: x, y data, unzipped
    """
    return np.array([item[0] for item in data]), np.array([item[1] for item in data])


def one_hot_encode(labels):
    """
    One hot encode labels

    :param labels: label vector, with numbers
    :return:  one hot encoded labels
    """
    new_labels = np.zeros((len(labels), len(CATEGORIES)))
    new_labels[np.arange(labels.size), labels] = 1
    return new_labels


def train(all_data):
    """
    Run training on data

    Reports different metrics after training

    :param all_data: input data
    """
    best_model = None

    folds = kfold(all_data)
    avg_epoch_data_train = EpochData()
    avg_epoch_data_val = EpochData()

    k = len(folds)
    for fold in range(k):
        # define the model
        if LOGISTIC:
            model = LogisticRegression(LEARNING_RATE, PRINCIPAL_COMPONENTS)
        else:
            model = SoftmaxRegression(LEARNING_RATE, PRINCIPAL_COMPONENTS, len(CATEGORIES))

        # split data
        val_data, test_data = split_x_y(folds[fold]), split_x_y(folds[(fold + 1) % k])
        train_data = None
        for i in range(k):
            if i != fold and i != ((fold + 1) % k):
                print(i)
                if train_data is None:
                    train_data = folds[i]
                else:
                    train_data = np.concatenate((train_data, folds[i]))
        train_data = split_x_y(train_data)

        print(train_data[0])
        pca = PCA(train_data[0], PRINCIPAL_COMPONENTS)

        # PCA and one_hot
        train_data, test_data, val_data = transform(pca, train_data), transform(pca, test_data), transform(pca,
                                                                                                           val_data)
        validation_performance = EpochData()
        training_performance = EpochData()

        assert not (any([val_img in train_data for val_img in val_data]))

        for epoch in range(EPOCHS):
            if STOCHASTIC_GRADIENT:
                model.stochastic_gradient_descent(train_data[0], train_data[1])
            else:
                model.batch_gradient_descent(train_data[0], train_data[1])

            train_prob = model.probabilities(train_data[0])
            val_prob = model.probabilities(val_data[0])

            training_error = model.loss(train_data[1], train_prob)
            validation_error = model.loss(val_data[1], val_prob)

            traning_acc = model.accuracy(train_prob, train_data[1])
            validation_acc = model.accuracy(val_prob, val_data[1])

            if epoch % 10 == 0:
                print("Training error: {}, validation error: {}, accuracy: {}".format(training_error, validation_error, traning_acc))

            # save
            validation_performance.save(validation_error, validation_acc)
            training_performance.save(training_error, traning_acc)

        # plot the graphs
        data_to_plot = [training_performance.error, validation_performance.error]
        legends = ["Training error", "Validation error"]
        #visualize_data(data_to_plot, legends, "Epoch", "Cross entropy error")

        data_to_plot = [training_performance.acc, validation_performance.acc]
        legends = ["Training accuracy", "Validation accuracy"]
        #visualize_data(data_to_plot, legends, "Epoch", "Accuracy")

        # save the validation data to the model
        model.epoch_data = validation_performance

        # save the pca
        model.pca = pca

        avg_epoch_data_train.add(training_performance)
        avg_epoch_data_val.add(validation_performance)

        # save the best model
        if best_model is None:
            best_model = model
        elif best_model.epoch_data.score() > model.epoch_data.score():
            best_model = model

    avg_epoch_data_train.align(10)
    avg_epoch_data_val.align(10)

    visualize_data_avg(avg_epoch_data_train, avg_epoch_data_val)
    if not LOGISTIC:
        best_model.visualize_weights(model.pca)
