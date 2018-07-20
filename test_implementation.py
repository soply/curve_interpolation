#imports
import time
import numpy as np
from implementation import reconstructCurve
from joblib import Parallel, delayed
import problem_factory.curve_classes as cc
from problem_factory.sample_synthetic_data import sample_fromClass



def error_test(original_curve, t0 , t1, Np,numberOfLevels,lam):
	
	"""
	Function that estimates the error between the original curve and the
	reconstructed curve.
	"""

	def isNotValidRun():
		return Np <= numberOfLevels

	if isNotValidRun():
		return


	dom,curve,coeff,x,tan,norm = sample_fromClass(t0,t1,original_curve,Np,0.2)
	ymin,means,z,L,N = reconstructCurve(t0,t1,x,tan,dom, Np,numberOfLevels,lam)


	"""
	Helping function that calculates the projection onto the average tangent
	vector of a given level set.
	"""

	def proj(point,i):
		p = [0,0]
		tan = np.array([(-1)*N[i][1],N[i][0]])
		inner_prod = tan[0]*(point[0]-ymin[0][i]) + tan[1]*(point[1]-ymin[1][i])
		
		p[0] = ymin[0][i] + inner_prod*tan[0]
		p[1] = ymin[1][i] + inner_prod*tan[1]

		return p

	
	"""
	Calculation of the error.
	"""
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
	
	print('Error value:', err,'with lambda =', lam,', number of levels=',
							numberOfLevels,', and number of partitions =',Np)

	return err



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
	ymin,means,z,L,N = reconstructCurve(t0,t1,x,tan,dom, Np,numberOfLevels,lam)

	fig, ax = handle_2D_plot()

	add_scattered_pointcloud_simple(curve,ax,color = 'g',dim=2)
	add_scattered_pointcloud_simple(ymin,ax,color = 'r', dim = 2)
	add_scattered_pointcloud_simple(means,ax,color = 'b', dim = 2)

	plt.show()



if __name__ == '__main__':

	"""
	Paralell error testing with circle piece as original curve.
	"""

	start_time = time.time()
	
	circle = cc.Circle_Piece_2D(2)

	lam = [0.1,1,10,1000]
	levels = [5,10,50,100]
	
	err = Parallel(n_jobs=4)(delayed(error_test) 
			(circle,0,1.4,50*lev,lev,l) for lev in levels for l in lam) 
	
	#error_test(circle,0,1.4,1000,100,1)
	
	print('time:', (time.time()-start_time))
