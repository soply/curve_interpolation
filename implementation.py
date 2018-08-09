"""
Implementation of an algorithm in order to reconstruct a surface from a
scattered set of points with orthogonal error.

Based on the Master's thesis of Johannes Piendl "Learning a Manifold from
Functional Data in Low Dimension". 
"""


# Imports
import numpy as np
import numpy.linalg as npl
import scipy.sparse.linalg as spl
from scipy import sparse, optimize
from sklearn.decomposition import PCA
from partitioning.based_on_domain import create_partitioning_n



def reconstructCurve(t0, t1, points,dom,N,numberOfLevels,D):
	
	"""
	Parameters
	==================

	t0, t1: float
		Start and endtime of the curve.

	N: integer
		Number of partitions
	
	points: np.array of floats, size D x numberOfPoints(N)
		Scattered points around the curve with orthogonal error.
	
	dom: np.array of floats, size numberOfPoints(N)
		Time points sampled in [t0,t1].

	numberOfLevels: integer
		Number of level sets.

	D: integer
		Dimension of the surface. 


	Returns
	==================
	
	opt_vec: np.array of floats, size D x numberOfLevels
		A set of points that approximate the curve.
	
	means: np.array of floats, size D x numberOfLevels
		The mean point of each level set.
		
	labels: np.array of floats, size numberOfLevels x numberOfPoints(N) x 2
		The level sets.

	N_list: list of matrices, size numberOfLevels
		The avarage normal plane/vector for each level set.
	
	t_list: list of vectors, size numberOfLevels
		The avarage tangent for each level set.

	"""




	"""
	Step 1: Creating the level sets and calculating the mean of each level set.
	"""

	means = np.zeros((D,numberOfLevels))

	labels = np.zeros(N).astype('int')
	labels, edges = create_partitioning_n(dom, numberOfLevels)
	
	for i in range(numberOfLevels):	
		means[:,i] = np.mean(points[:,labels == i+1], axis = 1)
	

		
	#Helping function, returns the number of points in a given level set.
	def getNumberOfPointsInLevel(i):
		return len(np.where(labels == i+1))


	
	
	"""
	Step 2: Finding the avarage normal vector for each level set.
	"""

	pca = PCA()
	N_list = []
	t_list = []

	for j in range(numberOfLevels):
		pca = pca.fit(points[:,labels == j+1].T)
		n = pca.components_[0:-1,:]
		t = pca.components_[-1,:]
		
		N_list.append(n)
		t_list.append(t)

	
	
	
	"""
	Step 3: Setting up the distance function and minimalizing it. 

	"""
	
	# Projection operators onto the normal and the tangential part of the curve
	P_par = np.zeros((numberOfLevels,D,D))
	P_ort = np.zeros((numberOfLevels,D,D))

	for j in range(numberOfLevels):
		P_ort[j] = np.outer(t_list[j],t_list[j])
		P_par[j] = np.identity(D) - P_ort[j]

	
	# The distanse function.
	def dist(y):
		s = 0
		for j in range(numberOfLevels):			
			s += npl.norm(np.matmul(P_ort[j],(y[D*j:D*j+D]-means[:,j])))**2 

		for j in range(numberOfLevels-1):
			s += npl.norm(np.matmul(P_par[j],
							(y[D*(j+1):D*(j+1)+D]-y[D*j:D*j+D])))**2

		return s

	
	n = numberOfLevels

	# Implementation of the linear system that arises when setting the gradient
	# of the distance function to zero.
	diag = np.zeros((n,D,D))
	upper_diag = np.zeros((n-1,D,D))
	lower_diag = np.zeros((n,D,D))	

	diag[0] = P_ort[0] + P_par[0]
	diag[n-1] = P_ort[n-1] + P_par[n-2]

	for j in range(1,n):
		diag[j] = P_ort[j] + P_par[j] + P_par[j-1]
		upper_diag[j-1] = -P_par[j-1]
		lower_diag[j-1] = -P_par[j-1]
	 
	lower_diag[n-1] = P_ort[n-1]


	A = sparse.bmat([[diag[i] if i == j 
						else upper_diag[j] if i-j == 1 
						else lower_diag[i] if i-j == (-1)  
						else None for i in range(n)] 
							for j in range(n)], format = 'csc')


	b = np.zeros(D*n)

	for j in range(1,n+1):
		temp1 = P_ort[j-1]
		b[D*j-D:D*j] = np.matmul(temp1,means[:,j-1])
	
	
	# Solving the linear system.
	ymin = spl.spsolve(A,b)	
		
	
	# Rewreiting the solution to a more usable form.
	y = np.zeros((D,numberOfLevels))

	for i in range(numberOfLevels):
		y[:,i] = ymin[D*i:D*i+D]

	

	
	"""
	The minimalization of the distance funtion is not uniqe, all solutions are
	on the for ymin + t*z, where z is a vector in the kernel of A. 
	
	Step 4: Finding the the optimal solution.
	"""

	Msum= np.zeros((D,D))
	msum = np.zeros(D)
	prod = np.eye(D)
	
	for l in range(numberOfLevels):
		prod = np.matmul(P_par[-(l+1)],prod)
		Msum += np.matmul(prod.T,prod)
		msum += np.matmul(prod.T,(means[:,-(l+1)] - y[:,-(l+1)]))
	
	M = np.matmul(N_list[-1],Msum)
	M = np.matmul(M,N_list[-1].T)
	m = np.matmul(N_list[-1],msum)

	u = npl.solve(M,m)

	
	z = np.zeros((D,numberOfLevels))

	for i in range(numberOfLevels-1):
		z[:,-(i+1)] = np.matmul(N_list[-(i+1)].T,u)
		u = np.matmul(N_list[-(i+2)],np.matmul(N_list[-(i+1)].T,u))
	

	z[:,0] = np.matmul(N_list[0].T,u)

	opt_vec = y + z



	
	# Returning the solution.	
	return opt_vec, means, labels, t_list

