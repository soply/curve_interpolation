#imports
import time
import sys
import os
import numpy as np
import implementation_D_dim as imp
import mls

from joblib import Parallel, delayed 
import matplotlib.pyplot as plt
import visualisation.vis_nD as vis
import problem_factory.curve_classes as cc
from problem_factory.sample_synthetic_data import sample_fromClass



def error_test(original_curve,t0,t1,Np,numberOfLevels,D,error,idx,rep,s):
	
	"""
	Function that estimates the error between the original curve and the
	reconstructed curve.
	"""

	def isNotValidRun():
		return Np <= numberOfLevels or  D < 0

	if isNotValidRun():
		return


	dom,curve,coeff,x,tan,norm = sample_fromClass(t0,t1,original_curve,Np,s)
	
	if (sys.argv[2] == 'mls') or (sys.argv[1] == 'mls'):
		ymin,means,labels,t = mls.reconstructCurve(t0,t1,x,dom,Np,numberOfLevels,D)
	else:
		ymin,means,labels,t = imp.reconstructCurve(t0,t1,x,dom,Np,numberOfLevels,D)
	

	"""
	Calculation of the error.
	"""
	err = 0
	count = 0

	for i in range(numberOfLevels):

		projections = (ymin[:,i] + \
					np.outer(t[i],(t[i])).dot((x[:,labels == i+1].T\
														- ymin[:,i]).T).T).T

		err += np.sum(np.square(np.linalg.norm(projections\
								- curve[:,labels == i+1], axis = 1)))

	err = 1.0/(x[0].size)*err
	
	error[idx,rep] = [err,Np,s]




def visual_test(original_curve, t0 , t1, Np,numberOfLevels,D):

	"""
	Fuction that plots the original curve, the means of each level
	set, and the reconstructed curve.
	"""

	def isNotValidRun():
		return Np <= numberOfLevels or D < 2 or D > 3 

	if isNotValidRun():
		return

	dom,curve,coeff,x,tan,norm = sample_fromClass(t0,t1,original_curve,Np,0.2)
	

	if (sys.argv[2] == 'mls') or (sys.argv[1] == 'mls'):
		ymin,means,labels,t = mls.reconstructCurve(t0,t1,x,dom,Np,numberOfLevels,D)
	else:
		ymin,means,labels,t = imp.reconstructCurve(t0,t1,x,dom,Np,numberOfLevels,D)
	

	if D == 2:
		fig, ax = vis.handle_2D_plot()
	elif D == 3:
		fig, ax = vis.handle_3D_plot()

	vis.add_scattered_pointcloud_simple(curve,ax,color = 'g',dim= D)
	vis.add_scattered_pointcloud_simple(ymin,ax,color = 'r', dim = D)
	vis.add_scattered_pointcloud_simple(means,ax,color = 'b', dim = D)

	plt.show()




def error_plot(error):

	"""
	Function that plots the esstimated error against the number of points for
	each sigma.
	"""

	index = len(error[:,0,0])
	err = np.zeros((index,3))

	for i in range(index):
		av_err = np.mean(error[i,:,0])
		err[i] = [av_err,error[i,0,1],error[i,0,2]]
	
	
	for s in sigma:
		i = np.where(err[:,2] == s)
		e = np.take(err,i,axis = 0)
		plt.plot(e[:,:,1][0],e[:,:,0][0], label = 'sigma: ' + str(s))
		plt.legend()

	plt.xlabel('Number of points')
	plt.ylabel('Error')
	plt.yscale('log')
	plt.xscale('log')

	plt.show()
		



if __name__ == '__main__':

	start_time = time.time()

	try:
		n_jobs = int(sys.argv[1])
	except:
		n_jobs = 4
	
	# test curves and test values.
	circle = cc.Circle_Piece_2D(2)
	helix = cc.Helix_Curve_3D(3)

	levels = [5,10,100,200]
	sigma = [0.01,0.1,0.5]


	# Creating a memory-map in order to store the errror measurments.
	folder = './joblib_memmap'

	try:
		os.mkdir(folder)
	except FileExistsError:
		pass

	rep = 10

	our_shape = np.zeros((len(sigma)*len(levels),rep,3))
	
	error = np.memmap(os.path.join(folder, 'err'),
				dtype='float64', shape=our_shape.shape,mode='w+')
	
	"""
	# Running the tests.
	Parallel(n_jobs=n_jobs)(delayed(error_test) 
			(helix,0,1.4,50*lev,lev,3,error,i*len(sigma)+k,r,s) 
									for i,lev in enumerate(levels)
											for k,s in enumerate(sigma)
														for r in range(rep))
	error_plot(error)
	
	
	Parallel(n_jobs=n_jobs)(delayed(error_test) 
			(circle,0,1.4,50*lev,lev,2,error,i*len(sigma)+k,r,s) 
									for i,lev in enumerate(levels)
											for k,s in enumerate(sigma)
														for r in range(rep))
	"""
	print(time.time()-start_time)

	visual_test(helix,0,1,1000,20,3)
	visual_test(circle,0,1,1000,20,2)

	#error_plot(error)

