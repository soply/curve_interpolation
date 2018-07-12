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


def test(original_curve, t0 , t1, N,numberOfLevels):
	dom,curve,coeff,x,tan,norm = sample_fromClass(t0, t1, original_curve,N,0.2)

	res = reconstructCurve(t0, t1, x, tan, dom, N,numberOfLevels)

	t = np.linspace(0,2,10)
	for i in range(numberOfLevels):
		print(res[i](t))
		plt.plot(res[i](t)[0],res[i](t)[1])
	
	fig, ax = handle_2D_plot()
	add_scattered_pointcloud_simple(curve,ax,color = 'g',dim=2)
	plt.show()
	
	

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
	
	res: list of linear functions that are orthogonal to the curve.
	

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
	means = np.zeros((numberOfLevels,2))

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
			means[i][0] = float(x_mean)/count 
			means[i][1] = float(y_mean)/count

	
	"""
	Step 2: creating the matrix P_j and finding Z_j
	"""

	matrix_list = []
	N_list = []
	Z_list = []

	for j in range(numberOfLevels):
		t = 0

		for point in L[j]:
			if point[0] != 0 or point[1] != 0:
				t = t+1

		
		shiftedPoints = np.zeros((t,2))
		P = np.zeros((t,2))


		for i in range(t):
			shiftedPoints[i][0] = L[j][i][0] - means[j][0]
			shiftedPoints[i][1] = L[j][i][1] - means[j][1]
		
			P[i][0] = shiftedPoints[i][0]
			P[i][1] = shiftedPoints[i][1]

		u,s,v = npl.svd(P) 
		n = v[0]
		
		z = lambda t: means[j] + np.array([t*n[0],t*n[1]]).T

		N_list.append(n)
		Z_list.append(z)
		matrix_list.append(P)
	

	"""
	Step 3: Set up the distance function and minimalizing it. By defoult mplemented with
	lambda = 1.

	"""
	
	def e(y):
		s = 0
		s += abs(y[0]-means[0][0]) + abs(y[1] -means[0][1])-((y[0]-means[0][0])*N_list[0][0] +(y[1]-means[0][1])*N_list[0][1])**2 + ((y[2]-y[0])*N_list[0][0] +(y[3]-y[1])*N_list[0][1])**2
		for j in range(1,numberOfLevels):
			s += abs(y[2*j]-means[j][0]) + abs(y[2*j+1] - means[j][1])-((y[2*j]-means[j][0])*N_list[j][0]+(y[2*j+1]-means[j][1])*N_list[j][1])**2 
		for j in range(1,numberOfLevels-1):
			s += lam*((y[2*j+2]-y[2*j])*N_list[j][0] + (y[2*j+3]-y[2*j+1])*N_list[j][1])**2 
		return s
	
	start = np.zeros(numberOfLevels*2)
	ymin = fmin(e,start)

	"""
	Step 4: returning the solutions.
	"""
	
	mu = np.zeros(numberOfLevels)
	z = np.zeros((numberOfLevels,2))

	mu[0] = 1

	for j in range(1,numberOfLevels):
		mu[j] = 1.0/(N_list[j-1][0]*N_list[j][0] +N_list[j-1][1]*N_list[j][1])*mu[j-1]
		z[j] = mu[j]*N_list[j]
	
	
	y = np.zeros((numberOfLevels,2))

	for i in range(numberOfLevels):
		y[i][0] = ymin[2*i]
		y[i][1] = ymin[2*i+1]
	
	res = []	
	print('ymin =', ymin)
	for i in range(numberOfLevels):
		f = lambda t: ymin[i] + Z_list[i](t)
		res.append(f)

	return res




circle = cc.Circle_Piece_2D(2)
test(circle,0,1.4,100,5)
