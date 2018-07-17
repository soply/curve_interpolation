# coding: utf8

"""
Implementation of an Algorithm in order to reconstruct at curve from a
scattered set of points with orthogonal error.

Based on the Master's thesis of Johannes Piendl "Learning a Manifold from
Functional Data in Low Dimension". 
"""

# Import stuff
import matplotlib.pyplot as plt  # For plotting
import numpy as np
import math
import random
import sys
import numpy.linalg as npl
from scipy.optimize import fmin
import pylab as pl

from partitioning.based_on_domain import (create_partitioning_dt,
                                          create_partitioning_n)
from performance.error_measurement import l2_error
from geometry_tools.utils import means_per_label
import problem_factory.curve_classes as cc
from problem_factory.sample_synthetic_data import sample_fromClass
from visualisation.vis_nD import *



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
		Time points in [t0,t1] where points are sampled.
		This parameter can be changed into the evaluation function (h) which we
		can use to construct the level sets.
		(Must be injective according to our assumptions).

	numberOfLevels: integer
		Number of levelsets.

	Returns
	==================
	
	y: np.array of floats, size 2 x M
		The minimalization of the distance function.
	
	means: np.array of floats, size 2 x M
		The mean points of each level set.

	z: np.array of floats, size M x 2 
		a vector that spans the set of solutions for the minimalization
		problem. All solutions are on the form y + t*z, where t is a real
		number. 
	
	L: np.array of floats, size M x numberOfPoints x 2
		The level sets.

	"""

	"""
	Step 1: Creating the level sets and calculating the mean of each level set.
	"""
	
	b0,b1 = t0,t1

	"""
	In case of an evaluation function (h), we set b0 = h_min, b1 = h_max.
	"""

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


	def getNumberOfPointsInLevel(i):	
		c = 0
		while L[i][points[0].size-1-c][0]==0 and L[i][points[0].size-1-c][1]==0:
			c += 1
		
		return points[0].size - c


	"""
	Step 2: creating the matrix P_j and finding Z_j
	"""

	matrix_list = []
	N_list = []
	Z_list = []
		

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
		
		z = [[means[0][j],means[1][j]],n]	
		
		N_list.append(n)
		Z_list.append(z)
		matrix_list.append(P)
	

	"""
	Step 3: Set up the distance function and minimalizing it. By defoult mplemented with
	lambda = 1.

	"""
	
	def e(y):
		s = 0
		for j in range(numberOfLevels):
			s += abs(y[2*j]-means[0][j])**2 + abs(y[2*j+1] - means[1][j])**2-((y[2*j]-means[0][j])*N_list[j][0]+(y[2*j+1]-means[1][j])*N_list[j][1])**2 
		for j in range(numberOfLevels-1):
			s += lam*((y[2*j+2]-y[2*j])*N_list[j][0] + (y[2*j+3]-y[2*j+1])*N_list[j][1])**2 
		return s

	
	A = np.zeros((2*numberOfLevels,2*numberOfLevels))
	n = numberOfLevels
	
	A[0:2,0:2] = np.identity(2) - np.outer(N_list[0],N_list[0]) +lam*np.outer(N_list[0],N_list[0])
	A[0:2,2:4] = -lam*np.outer(N_list[0],N_list[0])
	A[2*n-2:2*n,2*n-4:2*n-2] = -lam*np.outer(N_list[n-2],N_list[n-2])
	A[2*n-2:2*n,2*n-2:2*n] = np.identity(2) - np.outer(N_list[n-1],N_list[n-1]) + lam*np.outer(N_list[n-2],N_list[n-2])

	for j in range(2,n):
		A[2*j-2:2*j,2*j-2:2*j] = np.identity(2) -np.outer(N_list[j-1],N_list[j-1]) +lam*(np.outer(N_list[j-1],N_list[j-1]) +np.outer(N_list[j-2],N_list[j-2]))
		A[2*j-2:2*j,2*j:2*j+2] = -lam*np.outer(N_list[j-1],N_list[j-1])
		A[2*j-2:2*j,2*j-4:2*j-2] = -lam*np.outer(N_list[j-2],N_list[j-2])
	
	b = np.zeros(2*n)
	for j in range(1,n):
		temp1 = np.identity(2) - np.outer(N_list[j-1],N_list[j-1])
		temp2 = np.array([means[0][j-1],means[1][j-1]]) 
		b[2*j-2:2*j] = np.matmul(temp1,temp2)

	A_inv = npl.pinv(A)
	ymin = np.matmul(A_inv,b)
	
	
	"""
	Step 4: returning the solutions.
	"""
	
	mu = np.zeros(numberOfLevels)
	z = np.zeros((numberOfLevels,2))

	mu[0] = 1
	z[0] = N_list[0]

	for j in range(1,numberOfLevels):
		mu[j] = 1.0/(N_list[j-1][0]*N_list[j][0] +N_list[j-1][1]*N_list[j][1])*mu[j-1]
		z[j] = mu[j]*N_list[j]

	
	t_opt = 0
	denom = 0
	for i in range(numberOfLevels):
		prod = z[i][0]*ymin[2*i] + z[i][1]*ymin[2*i+1]

		temp = 0
		for k in range(getNumberOfPointsInLevel(i)):
			temp += z[i][0]*L[i][k][0] + z[i][1]*L[i][k][1]
		
		t_opt += prod - 1.0/(getNumberOfPointsInLevel(i))*temp
		denom += npl.norm(z[i])**2	

	
	t_opt = t_opt/(2-denom)  

	y = np.zeros((2,numberOfLevels))

	for i in range(numberOfLevels):
		y[0][i] = ymin[2*i] + t_opt*z[i][0]
		y[1][i] = ymin[2*i+1] + t_opt*z[i][1]

	return y, means, z, L, N_list



def test(original_curve, t0 , t1, N,numberOfLevels):

	dom,curve,coeff,x,tan,norm = sample_fromClass(t0, t1, original_curve,N,0.2)
	ymin,means,z,L,N = reconstructCurve(t0, t1, x, tan, dom, N,numberOfLevels)

	fig, ax = handle_2D_plot()

	add_scattered_pointcloud_simple(curve,ax,color = 'g',dim=2)
	add_scattered_pointcloud_simple(ymin,ax,color = 'r', dim = 2)
	add_scattered_pointcloud_simple(means,ax,color = 'b', dim = 2)

	"""
	Error estimation.
	"""

	def proj(point,i):
		p = [0,0]
		tan = np.array([(-1)*N[i][0],N[i][1]])
		inner_prod = tan[0]*(point[0]-ymin[0][i]) + tan[1]*(point[1]-ymin[1][i])
		
		p[0] = ymin[0][i] + inner_prod*tan[0]
		p[1] = ymin[1][i] + inner_prod*tan[1]

		return p


	err = 0
	count = 0

	for i in range(numberOfLevels):
		c = 0
		while L[i][x[0].size-1-c][0]==0 and L[i][x[0].size-1-c][1]==0:
			c += 1

		trunk_value = x[0].size - c

		for k in range(trunk_value):
			err += abs(curve[0][count] - proj(L[i][k],i)[0])**2 
			err += abs(curve[1][count] - proj(L[i][k],i)[1])**2 
			count += 1

	err = 1.0/(x[0].size)*err
	print('Error value:', err)
	
	plt.show()


circle = cc.Circle_Piece_2D(2)
test(circle,0,1.4,200,30)
