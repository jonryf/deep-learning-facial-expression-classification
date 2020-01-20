import numpy as np
from matplotlib import pyplot as plt

from Settings import CATEGORIES


class SoftmaxRegression:

    def __init__(self, lr, dim, c):
        """

        :param lr: Learning rate
        :param dim:  Number of dimensions (principal components)
        :param c: Number of categories
        """
        self.lr = lr
        self.c = c
        self.w = np.zeros((dim, c))

    def stochastic_gradient_descent(self, X, labels):
        """

        :param X: input data
        :param labels: labels for the input data
        """
        indices = [i for i in range(len(labels))]
        np.random.shuffle(indices)
        for i in indices:

            # make prediction
            data = X[i]
            label = labels[i]

            predicted = np.exp(data.dot(self.w)) / np.sum(np.exp(data.dot(self.w)), axis=0, keepdims=True)
            error = label - predicted
            # update weights
            grad = data.reshape(1, 10).T.dot(error.reshape(1, -1))
            self.w += self.lr * grad


    def probabilities(self, X):
        """

        :param X: input data
        :return: matrix, one row vector represent a probability vector for an image.
                Each column representing a class
        """
        return np.exp(X.dot(self.w)) / np.sum(np.exp(X.dot(self.w)), axis=1).reshape(len(X), 1)

    def batch_gradient_descent(self, X, labels):
        predicted = self.probabilities(X)
        error = labels - predicted
        grad = X.T.dot(error)
        self.w += 1/len(labels) * self.lr * grad

    def accuracy(self, prob_vec, labels):  # prob_vec row is probabilities of a single instance
        """

        :param prob_vec: a vector of the calculated probabilities
        :param labels: labels of the input data
        :return: the accuracy
        """
        numer = 0
        denom = len(labels)
        for i in range(len(labels)):
            y = list(prob_vec[i]).index(max(prob_vec[i]))
            t = list(labels[i]).index(max(labels[i]))
            if y == t:
                numer += 1
        accuracy = numer / denom
        return accuracy

    def visualize_weights(self, pca):
        """
        Visualize the weights as images

        :param pca: PCA instance
        """
        visualized = pca.components.dot(self.w).T.reshape((self.c, -1))
        imgs_data = [img.reshape(224, 192) for img in visualized]
        imgs_data = np.concatenate(imgs_data, axis=1)

        plt.title(" - ".join(CATEGORIES))
        plt.imshow(imgs_data, cmap="gray")
        plt.show()

    def loss(self, labels, predicted):
        """

        :param labels: input labels
        :param predicted: predicted values (as probability vectors)
        :return: cross entropy loss
        """
        score = 0
        for n in range(predicted.shape[0]):
            for category in range(self.c):
                score += labels[n, category] * np.log(predicted[n, category])

        return - score / predicted.shape[0]

