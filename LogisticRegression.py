import numpy as np
from Utils import sigmoid


class LogisticRegression:

    def __init__(self, lr, dim):
        """
        :param lr: learning rate
        :param dim: the number of the dimensions (principle components)
        """
        self.lr = lr
        self.w = np.zeros(dim)

    def stochastic_gradient_descent(self, X, labels):
        """
        :param X: the input data
        :param labels: the labels for the input data
        """
        indices = [i for i in range(len(labels))]
        np.random.shuffle(indices)
        for i in indices:

            # make prediction
            data = X[i]
            label = labels[i]
            predicted = sigmoid(data.dot(self.w))
            error = label - predicted

            # update weights
            for i in range(len(self.w)):
                grad = error * data[i]
                self.w[i] += self.lr * grad

    def probabilities(self, X):
        """

        :param X: the input data
        :return: a vector of probability. Each number in the vector represent one of the input
        images and is a number between 0 and 1.
        """
        return sigmoid(X.dot(self.w)).reshape(-1)

    def accuracy(self, prob_vec, labels):
        """

        :param prob_vec: a probability vector of all of the images
        :param labels: the labels for all of the images
        :return:
        """
        correct = np.round(prob_vec) == labels

        correct = np.sum(correct)
        accuracy = correct / len(labels)
        return accuracy

    def batch_gradient_descent(self, X, labels):
        """
        Calculate the gradient

        :param X: input data
        :param labels: labels for input data
        """
        predicted = self.probabilities(X)
        error = labels - predicted
        grad = X.T.dot(error)
        self.w += self.lr * grad

    def loss(self, labels, predicted):
        """
        Calculate the cross entropy loss

        :param labels: all of the labels
        :param predicted: the predicted value (probability)
        :return:
        """
        cost = -(np.log(predicted[labels == 1]).sum() + np.log(1 - predicted[~(labels == 1)]).sum()) / len(labels)
        return cost
