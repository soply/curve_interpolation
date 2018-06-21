# coding: utf8
import numpy as np


def l2_error(X_curve, X_curve_approx):
    """ Returns the Frobenius norm of X_curve - X_curve_approx. """
    return np.linalg.norm(X_curve - X_curve)
