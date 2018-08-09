#imports
import sys
import os
import numpy as np
import numpy.linalg as npl
import implementation as imp
import mls
from joblib import Parallel, delayed 
import matplotlib.pyplot as plt
import visualisation.vis_nD as vis
import problem_factory.curve_classes as cc
from problem_factory.sample_synthetic_data import sample_fromClass




def error_test(original_curve,t0,t1,Np,numberOfLevels,D,error,idx,rep,s,deg):
	
	"""
	Function that estimates the error between the original curve and the
	reconstructed curve.
	"""

	def isNotValidRun():
		return Np <= numberOfLevels or  D < 0

	if isNotValidRun():
		return


	dom,curve,coeff,x,tan,norm = sample_fromClass(t0,t1,original_curve,Np,s)	
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

		err += np.sum(np.square(npl.norm(projections\
									- curve[:,labels == i+1], axis = 1)))

	err = 1.0/(x[0].size)*err
	error[idx,rep] = [err,Np,s]





def mls_error_test(original_curve,t0,t1,Np,numberOfLevels,D,error,idx,rep,s,deg):
	
	"""
	Function that estimates the error between the original curve and the
	reconstructed curve for the mls algorithm.
	"""
		
	def isNotValidRun():
		return Np <= numberOfLevels or  D < 0

	if isNotValidRun():
		return

	
	dom,curve,coeff,x,tan,norm = sample_fromClass(t0,t1,original_curve,Np,s)
	poly_val,X,means=mls.reconstructCurve(t0,t1,x,dom,Np,numberOfLevels,D,deg)

	err = 0
	for i in range(Np):
		err += npl.norm(poly_val[:,i] - curve[:,i])**2

	error[idx,rep] = [err,Np,s]

		

		

def visual_test(original_curve, t0 , t1, Np,numberOfLevels,D,deg):

	"""
	Fuction that plots the original curve, the means of each level
	set, and the reconstructed curve.
	"""

	def isNotValidRun():
		return Np <= numberOfLevels or D < 2 or D > 3 

	if isNotValidRun():
		return

	dom,curve,coeff,x,tan,norm = sample_fromClass(t0,t1,original_curve,Np,0.2)
	
	if D == 2:
		fig, ax = vis.handle_2D_plot()
	elif D == 3:
		fig, ax = vis.handle_3D_plot()


	vis.add_scattered_pointcloud_simple(curve,ax,color = 'g',dim= D)

	if function == mls_error_test:
		p,X,means =mls.reconstructCurve(t0,t1,x,dom,Np,numberOfLevels,D,deg)
		vis.add_scattered_pointcloud_simple(p,ax,color = 'r', dim = D)
	else:
		ymin,means,labels,t = imp.reconstructCurve(t0,t1,x,dom,Np,numberOfLevels,D)
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


	# test curves and test values.
	circle = cc.Circle_Piece_2D(2)
	helix = cc.Helix_Curve_3D(3)

	levels = [10,100,200]
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
	
	

	# Reading user input.
	n_jobs = 4
	function = error_test

	for i in range(len(sys.argv)):
		if type(sys.argv[i]) == int:
			n_jobs = sys.argv[i]

		if sys.argv[i] == 'mls':
			function = mls_error_test
	


	# Running the tests.
	Parallel(n_jobs=n_jobs)(delayed(function) 
			(helix,0,1.4,50*lev,lev,3,error,i*len(sigma)+k,r,s,3) 
									for i,lev in enumerate(levels)
											for k,s in enumerate(sigma)
														for r in range(rep))
	error_plot(error)
	
	
	Parallel(n_jobs=n_jobs)(delayed(function) 
			(circle,0,1.4,50*lev,lev,2,error,i*len(sigma)+k,r,s,3) 
									for i,lev in enumerate(levels)
											for k,s in enumerate(sigma)
														for r in range(rep))
	error_plot(error)



	visual_test(helix,0,1,1000,10,3,3)
	visual_test(circle,0,1,1000,10,2,3)

