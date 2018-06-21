# coding: utf8
import numpy as np


def create_partitioning_n(t, n_partitions):
    """
    """
    N = t.shape[0]
    edges = np.linspace(np.min(t) - 1e-10, np.max(t) + 1e-10, n_partitions + 1)
    labels = np.digitize(t, edges)
    return labels, edges


def create_partitioning_dt(t, dt):
    """
    """
    N = t.shape[0]
    edges = np.arange(np.min(t) - 1e-10, np.max(t) + 1e-10, dt)
    labels = np.digitize(t, edges)
    return labels, edges
