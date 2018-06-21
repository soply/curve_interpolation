# coding: utf8
import numpy as np


def means_per_label(X, labels):
    """
    """
    D, N = X.shape
    J = len(set(labels)) # No of different labels
    means = np.zeros((D, J))
    for i, label in enumerate(set(labels)):
        means[:,i] = np.mean(X[:,labels == label], axis = 1)
    return means
