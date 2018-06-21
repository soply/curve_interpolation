# coding: utf8
import numpy as np


def create_partitioning_n(t, n_partitions):
    """
    Creates a partition of the t values equidistantly into n partitions.

    Parameters
    =========================
    t : np.array of floats, size N
        Time points in curve domain [t0, t1]

    n_partitions : integer
        Number of partitions

    Returns
    ==========================
    Returns the labels indicating the membership of each point, as well as the
    edges that form the partitioning of the t-domain.
    """
    N = t.shape[0]
    edges = np.linspace(np.min(t) - 1e-10, np.max(t) + 1e-10, n_partitions + 1)
    labels = np.digitize(t, edges)
    return labels, edges


def create_partitioning_dt(t, dt):
    """
    Creates a partition of the t values equidistantly into n partitions.

    Parameters
    =========================
    t : np.array of floats, size N
        Time points in curve domain [t0, t1]

    dt : float
        Width of the partitions in curve domain [t0, t1]

    Returns
    ==========================
    Returns the labels indicating the membership of each point, as well as the
    edges that form the partitioning of the t-domain.
    """
    N = t.shape[0]
    edges = np.arange(np.min(t) - 1e-10, np.max(t) + 1e-10, dt)
    labels = np.digitize(t, edges)
    return labels, edges
