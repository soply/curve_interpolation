#imports
import numpy as np
from sklearn.decomposition import PCA
from partitioning.based_on_domain import create_partitioning_n
import numpy.linalg as npl

def reconstructCurve(t0,t1,points,dom,N,numberOfLevels,D):
	

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
	
	

	solution = np.zeros((D,numberOfLevels))

	for k in range(numberOfLevels):
		points_in_level = points[:,labels == k+1]
		
		x = np.zeros(len(points_in_level[0]))		

		for j in range(len(points_in_level[0])):
			x[j] = np.dot(tan[:,k],points_in_level[:,j])
		

		#zero_value = 0
		zero_value = np.zeros(D)
		for i in range(D):
			poly = np.polyfit(x,points_in_level[i,:],D-1)
			#zero_value += poly[-1]
			zero_value[i] = poly[-1] 
		
		P_ort =  np.outer(tan[:,k],tan[:,k])

		# P(r) = q + p(0)*a
		solution[:,k] = np.dot(P_ort,means[:,k]) #+ zero_value
	

	return solution,means, labels, tan


