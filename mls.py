"""
Implementation of an algorithm in order to reconstruct a surface from a
scattered set of points with orthogonal error.

Based on the article "Mesh-Independent Surface Interpolation" by David Levin.
"""


#imports
import numpy as np
from sklearn.decomposition import PCA
from partitioning.based_on_domain import create_partitioning_n
import numpy.linalg as npl



def reconstructCurve(t0,t1,points,dom,N,numberOfLevels,D,deg):
	
	"""
	Parameters
	==================

	t0, t1: float
		Start and endtime of the curve.

	N: integer
		Number of partitions
	
	points: np.array of floats, size 2 x M
		Scattered points around the curve with orthogonal error.
	
	dom: np.array of floats, size N
		Time points sampled in [t0,t1].

	numberOfLevels: integer
		Number of level sets.

	D: integer
		Dimension of the surface. 


	Returns
	==================

	poly_val: np.array of floats, size D x numberOfPoints(N)
		The evaluation of the estimated polynomial at each point.

	X: np.array of floats, size numberOfPoints(N)
		The projection of each point onto the tangent of the corresponding 
		level set.

	means: np.array of floats, size D x numberOfLevels
		The mean point of each level set.

	"""


	
	
	"""
	Step 1: Creating the level sets and calculating the mean of each level set.
	"""

	means = np.zeros((D,numberOfLevels))

	labels = np.zeros(N).astype('int')
	labels, edges = create_partitioning_n(dom, numberOfLevels)
	
	for i in range(numberOfLevels):	
		means[:,i] = np.mean(points[:,labels == i+1], axis = 1)
	


	
	"""
	Step 2: Finding the avarage tangent vector for each level set.
	"""

	pca = PCA()
	tan = np.zeros((D,numberOfLevels))

	for j in range(numberOfLevels):
		pca = pca.fit(points[:,labels == j+1].T)
		t = pca.components_[-1,:]
		
		tan[:,j] = t
	

	
	
	"""
	Step 3: Finding a polynomial that estimates the curve for each level set.
	"""

	X = np.zeros(N)
	poly_val = np.zeros((D,N))

	for k in range(numberOfLevels):		
		points_in_level = points[:,labels == k+1]
		n = len(points_in_level[0])
		x = np.zeros(n)

		for j in range(n):
			x[j] = np.dot(tan[:,k],points_in_level[:,j])
			X[k*n + j] = x[j]
		
		poly = np.polyfit(x,points_in_level.T,deg)
	
		for i in range(D):				
			poly_val[i][k*n:k*n+n] = np.polyval(poly[:,i],x)




	# Returning the solution.
	return poly_val,X,means


