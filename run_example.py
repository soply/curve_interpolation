# coding: utf8

"""
Simple example of how to run the code.
"""

# Import stuff
import numpy as np
import matplotlib.pyplot as plt # For plotting

from problem_factory.curve_classes import Circle_Piece_2D
from problem_factory.sample_synthetic_data import sample_fromClass
from visualisation.vis_nD import *


if __name__ == "__main__":
    # Code below is excuted when file is called via python run_example.py
    # Define some parameters
    N = 1000 # number of samples
    D = 3 # ambient dimension
    t0 = 0 # start point
    t1 = np.pi/2 # end point
    sigma = 0.1 # Normal detachment width
    """ Step 1: Creating synthetic data set. """
    # Create curve
    curve = Circle_Piece_2D(D)
    # Sample points
    t, X_curve, _, X, tan, nor = sample_fromClass(t0, t1, curve, N, sigma, tubetype = 'l2')
    # Visualize points
    fig, ax = handle_2D_plot()
    add_scattered_pointcloud_simple(X, ax, color = 'b', dim = 2)
    plt.show()
    """ Step 2: Splitting the data set into subsets and obtaining the means. """
    
