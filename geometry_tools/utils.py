# coding: utf8
import numpy as np


def means_per_label(X, labels):
    """
    Calculates the means of all subsets of X. Subset membership is indicated by
    labels.

    Parameters
    =================
    X : np.array of floats, size D x N
        Data points

    labels : np.array of ints, size N
        Indicates the membership of a data point.

    Returns
    =================
    means : np.array of floats, size D x #different labels
        Means of the data points belonging to a group, for each group.
    """
    D, N = X.shape
    J = len(set(labels)) # No of different labels
    means = np.zeros((D, J))
    for i, label in enumerate(set(labels)):
        means[:,i] = np.mean(X[:,labels == label], axis = 1)
    return means
