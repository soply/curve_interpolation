# coding: utf8
import numpy as np

"""
Methods to sample data from synthethic data.
"""

def sample_fromClass(t0, t1, curve, N, sigma, tubetype = 'l2'):
    """
    Sample N data points from a distribution evolving around the given curve
    between time points t0 and t1.

    Parameters
    ====================
    t0, t1 : float
        Start and endtime of the domain of the curve

    curve : Curve object as implemented in 'manifolds_as_curves_classes.py'.
        The actual curve, implementing shape, tangents, normals and so on.

    N : integer
        Number of samples

    sigma : real
        Width of the tube around the curve. Exact meaning depends on the tubetype.

    tubetype : str with the following options
        - 'l2' : The coefficients of the detachment vector are sampled uniformly
        from the l2 ball of radius sigma.
        -'linfinity' : The coefficients of the detachment vector are sampled
        uniformly from the linf ball of radius sigma.

    Returns
    ===================
    p_curvedomain : np.array of floats, size N
        Time points in [t0,t1] where points are sampled.

    p_curve : np.array of floats, size D x N
        Points curve(p_curvedomain) that are exactly on the curve.

    p_coeff_all : np.array of floats, size D x N
        Coefficients of the points. First row contains p_curvedomain values,
        rest contain basis coefficients for the detachment.

    X : np.array of floats, size D x N
        Samples p_curve with added normal detachment.

    tangentspaces : np.array of floats, size D x 1 x N
        Tangent vectors of the curve at p_curvedomain.

    normalspaces : np.array of floats, size D x D-1 x N
        Normal spaces of the curve at p_curvedomain.
    """
    # p_curvedomain = np.random.uniform(low = t0, high = t1, size = (N))
    p_curvedomain = np.linspace(t0, t1, N)
    p_curvedomain = np.sort(p_curvedomain)
    D = curve.get_n_features()
    # Containers
    p_curve = np.zeros((D, N))
    X = np.zeros((D, N))
    p_coeff_all = np.zeros((D, N)) # Contains t + normal coefficients
    p_coeff_all[0,:] = p_curvedomain
    tangentspaces = np.zeros((D, 1, N))
    normalspaces = np.zeros((D, D - 1, N))
    # Sample Detachment Coefficients
    if tubetype == 'linfinity':
        # Sample detachment coefficients from ||k||_inf < sigma.
        random_coefficients = np.random.uniform(-sigma, sigma,
                                            size = (D - 1, N))
    elif tubetype == 'l2':
        # Sample detachment coefficients from ||k||_2 < sigma
        rand_sphere = np.random.normal(size = (D - 1, N))
        rand_sphere = rand_sphere/np.linalg.norm(rand_sphere, axis = 0)
        radii = np.random.uniform(0, 1, size = N)
        radii = sigma * np.power(radii, 1.0/(D - 1))
        random_coefficients = rand_sphere * radii
    p_coeff_all[1:,:] = random_coefficients
    for i in range(N):
            p_curve[:,i] = curve.get_basepoint(p_curvedomain[i])
            tangentspaces[:,0,i] = curve.get_tangent(p_curvedomain[i])
            normalspaces[:,:,i] = curve.get_normal(p_curvedomain[i])
            normal_vector = np.sum(normalspaces[:,:,i] * random_coefficients[:,i],
                                   axis=1)
            X[:,i] = p_curve[:,i] + normal_vector
    return p_curvedomain, p_curve, p_coeff_all, X, tangentspaces, normalspaces
