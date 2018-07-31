#imports
import time
import sys
import os
import numpy as np
from implementation_D_dim import reconstructCurve
from joblib import Parallel, delayed 
import matplotlib.pyplot as plt
import visualisation.vis_nD as vis
import problem_factory.curve_classes as cc
from problem_factory.sample_synthetic_data import sample_fromClass



def error_test(original_curve,t0,t1,Np,numberOfLevels,D,lam,error,idx,rep):
	
	"""
	Function that estimates the error between the original curve and the
	reconstructed curve.
	"""

	def isNotValidRun():
		return Np <= numberOfLevels or  D < 0

	if isNotValidRun():
		return


	dom,curve,coeff,x,tan,norm = sample_fromClass(t0,t1,original_curve,Np,0.2)
	ymin,means,labels,N,t = reconstructCurve(t0,t1,x,tan,dom,
													Np,numberOfLevels,D,lam)

	
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
	
	error[idx,rep] = [err,Np,lam]



def visual_test(original_curve, t0 , t1, Np,numberOfLevels,D,lam):

	"""
	Fuction that plots the original curve, the means of each level
	set and the reconstructed curve.
	"""

	def isNotValidRun():
		return Np <= numberOfLevels or D < 2 or D > 3 

	if isNotValidRun():
		return

	dom,curve,coeff,x,tan,norm = sample_fromClass(t0,t1,original_curve,Np,0.2)
	ymin,means, labels,N,t = reconstructCurve(t0,t1,x,tan,dom,
													Np,numberOfLevels,D,lam)
	
	if D == 2:
		fig, ax = vis.handle_2D_plot()
	elif D == 3:
		fig, ax = vis.handle_3D_plot()

	vis.add_scattered_pointcloud_simple(curve,ax,color = 'g',dim= D)
	vis.add_scattered_pointcloud_simple(ymin,ax,color = 'r', dim = D)
	vis.add_scattered_pointcloud_simple(means,ax,color = 'b', dim = D)

	plt.show()



def error_plot(error):

	index = len(error[:,0,0])	
	err = np.zeros((index,3))

	for i in range(index):
		av_err = np.mean(error[i,:,0])
		err[i] = [av_err,error[i,0,1],error[i,0,2]]
	
	
	for l in lam:
		i = np.where(err[:,2] == l)
		e =np.take(err,i, axis = 0)
		plt.plot(e[:,:,1][0],e[:,:,0][0],label = 'lambda: ' + str(l))
		plt.legend()

	plt.xlabel('Number of points')
	plt.ylabel('Error')
	plt.yscale('log')
	plt.xscale('log')

	plt.show()
		


if __name__ == '__main__':

	"""
	Paralell error testing with circle piece as original curve.
	"""

	start_time = time.time()
	
	circle = cc.Circle_Piece_2D(2)
	helix = cc.Helix_Curve_3D(3)


	lam = [1,10,1000,100000]
	levels = [5,10,20,50,100,200]
	
	folder = './joblib_memmap'

	try:
		os.mkdir(folder)
	except FileExistsError:
		pass

	rep = 10

	our_shape = np.zeros((len(lam)*len(levels),rep,3))

	error = np.memmap(os.path.join(folder, 'err'),
				dtype='float64', shape=our_shape.shape,mode='w+')
	
	Parallel(n_jobs=2)(delayed(error_test) 
			(helix,0,1.4,50*lev,lev,3,l,error,i*len(lam)+k,r) 
								for i,lev in enumerate(levels) 
										for k,l in enumerate(lam) 
												for r in range(rep))
	error_plot(error)

	Parallel(n_jobs=2)(delayed(error_test) 
			(circle,0,1.4,50*lev,lev,2,l,error,i*len(lam)+k,r) 
								for i,lev in enumerate(levels) 
										for k,l in enumerate(lam) 
												for r in range(rep))
	print(time.time()-start_time)

	error_plot(error)
	visual_test(helix, 0,1,1000,20,3,1)
