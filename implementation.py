"""
Implementation of an Algorithm in order to reconstruct at curve from a
scattered set of points with orthogonal error.

Based on the Master's thesis of Johannes Piendl "Learning a Manifold from
Functional Data in Low Dimension". 
"""


# Imports
import matplotlib.pyplot as plt 
import numpy as np
import numpy.linalg as npl
import pylab as pl
import scipy.sparse.linalg as spl
from scipy import sparse



def reconstructCurve(t0, t1, points, diffpoints, dom, N,numberOfLevels, lam=1):
	
	"""
	Parameters
	==================

	t0, t1: float
		Start and endtime of the curve.

	N: integer
		Number of partitions
	
	points: np.array of floats, size 2 x M
		Scattered points around the curve with orthogonal error.
	
	diffpoints: np.array of floats, size 2 x M
		The values of the derivatives of the curve at each partition.

	dom: np.array of floats, size N
		Time points sampled in [t0,t1].

	numberOfLevels: integer
		Number of level sets.


	Returns
	==================
	
	y: np.array of floats, size 2 x M
		The optimal minimalization of the distance function.
	
	means: np.array of floats, size 2 x M
		The mean points of each level set.

	z: np.array of floats, size M x 2 
		a vector that spans the set of solutions for the minimalization
		problem. All solutions are on the form y + t*z, where t is a real
		number. 
	
	L: np.array of floats, size M x numberOfPoints x 2
		The level sets.

	N_list: list of vectors, size M
		The avarage normal vector for each level set.

	"""


	"""
	Step 1: Creating the level sets and calculating the mean of each level set.
	"""

	b0,b1 = t0,t1

	b = np.linspace(b0,b1,numberOfLevels + 1)
	L = np.zeros((numberOfLevels,points[0].size,2))
	means = np.zeros((2,numberOfLevels))

	for i in range(numberOfLevels):
		count = 0
		for p in range(N):
			if b[i] <= dom[p] and b[i+1] > dom[p]:
				L[i][count] = [points[0][p],points[1][p]]
				count = count +1
		
		x_mean,y_mean = 0,0
		for k in range(count):
			x_mean += L[i][k][0]
			y_mean += L[i][k][1]
		
		
		if count != 0:
			means[0][i] = float(x_mean)/count 
			means[1][i] = float(y_mean)/count



	"""	
	Helping function, returns the number of points in a given level set.
	"""
	

	def getNumberOfPointsInLevel(i):	
		c = 0
		while L[i][points[0].size-1-c][0]==0 and L[i][points[0].size-1-c][1]==0:
			c += 1
	
		return points[0].size - c



	"""
	Step 2: Finding the avarage normal vector for each level set.
	"""

	N_list = []		

	for j in range(numberOfLevels):
		
		shiftedPoints = np.zeros((getNumberOfPointsInLevel(j),2))
		P = np.zeros((getNumberOfPointsInLevel(j),2))

		for i in range(getNumberOfPointsInLevel(j)):
			shiftedPoints[i][0] = L[j][i][0] - means[0][j]
			shiftedPoints[i][1] = L[j][i][1] - means[1][j]
		
			P[i][0] = shiftedPoints[i][0]
			P[i][1] = shiftedPoints[i][1]

		u,s,v = npl.svd(P) 
		n = v[0]	
		
		N_list.append(n)
	

	
	"""
	Step 3: Setting up the distance function and minimalizing it. 

	"""

	def dist(y):
		s = 0
		for j in range(numberOfLevels):
			s += abs(y[2*j]-means[0][j])**2 + abs(y[2*j+1] - means[1][j])**2\
				-((y[2*j]-means[0][j])*N_list[j][0]\
				+(y[2*j+1]-means[1][j])*N_list[j][1])**2 
		for j in range(numberOfLevels-1):
			s += lam*((y[2*j+2]-y[2*j])*N_list[j][0]\
				+ (y[2*j+3]-y[2*j+1])*N_list[j][1])**2 
		return s


	n = numberOfLevels

	diag = np.zeros((n,2,2))
	upper_diag = np.zeros((n-1,2,2))
	lower_diag = np.zeros((n-1,2,2))	

	diag[0] = np.identity(2) - np.outer(N_list[0],N_list[0])\
				+lam*np.outer(N_list[0],N_list[0])
	diag[n-1] = np.identity(2) - np.outer(N_list[n-1],N_list[n-1])\
				+lam*np.outer(N_list[n-2],N_list[n-2])

	for j in range(1,n):
		diag[j] = np.identity(2)-np.outer(N_list[j],N_list[j])\
					+lam*(np.outer(N_list[j],N_list[j])\
					+np.outer(N_list[j-1],N_list[j-1]))
	for j in range(n-1):
		upper_diag[j] = -lam*np.outer(N_list[j],N_list[j])
		lower_diag[j] = -lam*np.outer(N_list[j],N_list[j])
	 

	A = sparse.bmat([[diag[i] if i == j 
						else upper_diag[j] if i-j == 1 
						else lower_diag[i] if i-j == (-1)  
						else None for i in range(n)] 
							for j in range(n)], format = 'csc')


	b = np.zeros(2*n)

	for j in range(1,n+1):
		temp1 = np.identity(2) - np.outer(N_list[j-1],N_list[j-1])
		temp2 = np.array([means[0][j-1],means[1][j-1]]) 
		b[2*j-2:2*j] = np.matmul(temp1,temp2)
	

	ymin = spl.spsolve(A,b)
	


	"""
	The minimalization of the distance funtion is not uniqe, all solutions are
	on the for ymin + t*z, where z is a vector in the kernel of A. 
	
	Settin up a vector that is in the kernel of A.
	"""
	
	mu = np.zeros(numberOfLevels)
	z = np.zeros((numberOfLevels,2))

	mu[0] = 1
	z[0] = N_list[0]

	for j in range(1,numberOfLevels):
		mu[j] = 1.0/(N_list[j-1][0]*N_list[j][0]\
		 			+ N_list[j-1][1]*N_list[j][1])*mu[j-1]
		z[j] = mu[j]*N_list[j]


	
	"""
	Finding the optimalized t-value.
	"""

	t_opt = 0
	denom = 0
	for i in range(numberOfLevels):

		prod = z[i][0]*ymin[2*i] + z[i][1]*ymin[2*i+1]

		temp = 0
		for k in range(getNumberOfPointsInLevel(i)):
			temp += z[i][0]*L[i][k][0] + z[i][1]*L[i][k][1]
		
		t_opt += 1.0/(getNumberOfPointsInLevel(i))*temp - prod
		denom += npl.norm(z[i])**2	
	
	t_opt = t_opt/denom


	
	"""
	Returning the solution.
	"""

	y = np.zeros((2,numberOfLevels))

	for i in range(numberOfLevels):
		y[0][i] = ymin[2*i] + t_opt*z[i][0]
		y[1][i] = ymin[2*i+1] + t_opt*z[i][1]

	return y, means, z, L, N_list



