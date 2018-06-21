# coding: utf8

"""
Simple example of how to run the code.
"""
# Import stuff
import matplotlib.pyplot as plt  # For plotting
import numpy as np

from partitioning.based_on_domain import (create_partitioning_dt,
                                          create_partitioning_n)
from geometry_tools.utils import means_per_label
from problem_factory.curve_classes import Circle_Piece_2D
from problem_factory.sample_synthetic_data import sample_fromClass
from visualisation.vis_nD import *

if __name__ == "__main__":
    # Code below is excuted when file is called via python run_example.py
    # Define some parameters
    N = 200 # number of samples
    D = 3 # ambient dimension
    t0 = 0 # start point
    t1 = np.pi/2 # end point
    sigma = 0.1 # Normal detachment width
    """ Step 1: Creating synthetic data set. """
    # Create curve
    curve = Circle_Piece_2D(D)
    # Sample points
    t, X_curve, _, X, tan, nor = sample_fromClass(t0, t1, curve, N, sigma, tubetype = 'l2')
    # Visualize points (visualisation in 2D means here we just look at the first 2
    # coordinates where the interesting stuff is happening in this example).
    fig, ax = handle_2D_plot()
    add_scattered_pointcloud_simple(X, ax, color = 'b', dim = 2) # Add X in blue
    add_scattered_pointcloud_simple(X_curve, ax, color = 'g', dim = 2) # Add curve(t) in green
    plt.show()
    # Visualize points and tangents
    fig2, ax2 = handle_2D_plot()
    add_scattered_pointcloud_simple(X, ax2, color = 'b', dim = 2)
    for i in range(N):
        add_affine_space(X[:,i], tan[:,0,i], 0.01, ax2, dim = 2)
    plt.show()
    """ Step 2: Splitting the data set into subsets and obtaining the means. """
    n_partitions = 10
    labels, edges = create_partitioning_n(t, n_partitions)
    # Get means
    means = means_per_label(X, labels)
    # Visualize the labels and means
    fig3, ax3 = handle_2D_plot()
    add_scattered_pointcloud(X, labels, ax3, labels, dim = 2)
    add_scattered_pointcloud_simple(means, ax3, color = 'm', dim = 2)
    plt.show()
