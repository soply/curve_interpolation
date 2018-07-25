#imports
import time
import sys
import os
import numpy as np
from implementation import reconstructCurve
from joblib import Parallel, delayed 
import matplotlib.pyplot as plt
import problem_factory.curve_classes as cc
from problem_factory.sample_synthetic_data import sample_fromClass



def error_test(original_curve, t0 , t1, Np,numberOfLevels,lam,error,idx):
	
	"""
	Function that estimates the error between the original curve and the
	reconstructed curve.
	"""

	def isNotValidRun():
		return Np <= numberOfLevels

	if isNotValidRun():
		return


	dom,curve,coeff,x,tan,norm = sample_fromClass(t0,t1,original_curve,Np,0.2)
	ymin,means,z,labels,N = reconstructCurve(t0,t1,x,tan,dom, Np,numberOfLevels,lam)


	tangents_per_levelset = np.zeros((2,numberOfLevels))

	for i in range(numberOfLevels):
		tangents_per_levelset[:,i] = np.array([(-1)*N[i][1],N[i][0]])

	
	"""
	Calculation of the error.
	"""
	err = 0
	count = 0

	for i in range(numberOfLevels):

		projections = (ymin[:,i] + \
					np.outer(tangents_per_levelset[:,i],
					(tangents_per_levelset[:,i])).dot((x[:,labels == i+1].T\
														- ymin[:,i]).T).T).T

		err += np.sum(np.square(np.linalg.norm(projections\
								- curve[:,labels == i+1], axis = 1)))

	err = 1.0/(x[0].size)*err
	
	error[idx] = [err,Np,lam]



def visual_test(original_curve, t0 , t1, Np,numberOfLevels,lam):

	"""
	Fuction that plots the original curve, the means of each level
	set and the reconstructed curve.
	"""

	def isNotValidRun():
		return Np <= numberOfLevels

	if isNotValidRun():
		return


	dom,curve,coeff,x,tan,norm = sample_fromClass(t0,t1,original_curve,Np,0.2)
	ymin,means,z,labels,N = reconstructCurve(t0,t1,x,tan,dom, Np,numberOfLevels,lam)

	fig, ax = handle_2D_plot()

	add_scattered_pointcloud_simple(curve,ax,color = 'g',dim=2)
	add_scattered_pointcloud_simple(ymin,ax,color = 'r', dim = 2)
	add_scattered_pointcloud_simple(means,ax,color = 'b', dim = 2)

	plt.show()



def error_plot(error):

	plt.plot(error[:,1],error[:,0])
	plt.xlabel('Number of points')
	plt.ylabel('Error')
	plt.yscale('log')

	plt.show()
		


if __name__ == '__main__':

	"""
	Paralell error testing with circle piece as original curve.
	"""

	start_time = time.time()
	
	circle = cc.Circle_Piece_2D(2)

	lam = [0.1,1,10,1000]
	levels = [5,10,50,100]
	
	folder = './joblib_memmap'

	try:
		os.mkdir(folder)
	except FileExistsError:
		pass

	our_shape = np.zeros((len(lam)*len(levels),3))

	error = np.memmap(os.path.join(folder, 'err'),
				dtype='float64', shape=our_shape.shape,mode='w+')
	
	Parallel(n_jobs=2)(delayed(error_test) 
			(circle,0,1.4,50*lev,lev,l,error,i*len(lam)+k) 
				for i,lev in enumerate(levels) for k,l in enumerate(lam))

	print(time.time()-start_time)
	error_plot(error)

