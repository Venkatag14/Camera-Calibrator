import numpy as np
import cv2
import matplotlib.pyplot as plt 
import copy
import os
import scipy.optimize as optimize
import math
from os import listdir
from os.path import dirname, abspath
from skimage.io import imread, imshow
from skimage import transform

def LoadImagesFromFolder(folder):
	images = []
	for file in listdir(folder):
		tmp = cv2.imread(folder + "\\" + file)
		if tmp is not None:
			images.append(tmp)
	return images

def ChessCornersLocal(images, dims, result_path):
	Corners = []
	count = 1
	for img in images:
		img = copy.deepcopy(img)
		ret, corners = cv2.findChessboardCorners(img, dims, None)
		if ret == True:
			CImg = cv2.drawChessboardCorners(img, dims, corners, ret)
			cv2.imwrite(result_path + "\\" + f"OrigCorner{count}.png", CImg)
			corners = np.array(corners, np.float32)
			corners = corners.squeeze(1)
			Corners.append(corners)
			count += 1
	Corners = np.array(Corners)
	return Corners

def ChessCornersGlobal(dims, len):
	
    Corners = []
    for i in range(1, dims[1]+1):
        for j in range(1, dims[0]+1):
            Corners.append((i*len, j*len))
    Corners = np.array(Corners, np.float32)
    return Corners

def Homography(LocalCorners, GlobalCorners):
	H = []
	for i in range(len(LocalCorners)):
		A = []
		for j in range(len(GlobalCorners)):
			xs, ys = GlobalCorners[j][0], GlobalCorners[j][1]
			xd, yd = LocalCorners[i][j][0], LocalCorners[i][j][1]
			a = [[xs, ys, 1, 0, 0, 0, -xd*xs, -xd*ys, -xd],
                [0, 0, 0, xs, ys, 1, -yd*xs, -yd*ys, -yd]]
			A.append(a)
		A = np.array(A)
		A = A.reshape(108,9)
		eigenvalues, eigenvectors = np.linalg.eig(np.dot(A.T, A))
        #finding eigen vector for minimum eigen value
		min_eigenvalue_index = np.argmin(eigenvalues)
		min_eigenvector = eigenvectors[:, min_eigenvalue_index]
        #assigning h_matrix
		min_eigenvector = min_eigenvector/min_eigenvector[8]
		h_matrix = min_eigenvector.reshape(3,3)
		H.append(h_matrix)
	return H

def v_matrix(H,i,j):
	i = i-1
	j = j-1
	H = H.T
	v = [H[i][0]*H[j][0], H[i][0]*H[j][1]+H[i][1]*H[j][0], H[i][1]*H[j][1], H[i][2]*H[j][0]+H[i][0]*H[j][2], H[i][2]*H[j][1]+H[i][1]*H[j][2],H[i][2]*H[j][2]]
	return v


def ClosedFormSolution(H_list):
	V = []
	for H in H_list:
		v12 = v_matrix(H,1,2)
		v11 = v_matrix(H,1,1)
		v22 = v_matrix(H,2,2)
		v11_v22 = []
		for i in range(6):
			v11_v22.append(v11[i]-v22[i])
		v = [[v12],
	         [v11_v22]]
		V.append(v)
	V = np.array(V)
	# print(V)
	V = V.reshape(26,6)
	# print(V)
	eigenvalues, eigenvectors = np.linalg.eig(np.dot(V.T, V))
	min_eigenvalue_index = np.argmin(eigenvalues)
	min_eigenvector = eigenvectors[:, min_eigenvalue_index]
    #assigning b
	b = min_eigenvector
	return b

def IntrinsicParams(b):
	alpha = math.sqrt(1/b[0])
	beta = math.sqrt(1/b[2])
	u0 = (-b[3]*(alpha**2)*beta)/(beta)
	v0 = -b[4]*(beta**2)
	return alpha, beta, u0, v0

def HomogenousMatrix(K, H_list):
	Kinv = np.linalg.inv(K)
	Rt = []
	for H in H_list:
		rt = np.dot(Kinv, H)
		Rt.append(rt)
	return Rt

def LossFunction(fx, fy, cx, cy, k1, k2, Transformation_Matrices, LocalCorners, GlobalCorners):
	# Transformation_Matrices = HomogenousMatrix(K, H_list)
	K = [[fx, 0, cx],[0, fy, cy],[0, 0, 1]]
	K = np.array(K, np.float32)
	Loss = 0
	for i in range(LocalCorners.shape[0]):

		Rt = Transformation_Matrices[i]
		for j in range(LocalCorners.shape[1]):
			# Rt = np.zeros((3,3))
			# Rt[:,0] = transformation_Matrix[:3,0]
			# Rt[:,1] = transformation_Matrix[:3,1]
			# Rt[:,2] = transformation_Matrix[:3,3]
			GlobalCorner = GlobalCorners[j]
			M = [GlobalCorner[0], GlobalCorner[1], 1]
			M = np.array(M)
			M = M.reshape(3,1)
			LocalCorner = LocalCorners[i][j]
			m = np.array([LocalCorner[0],
                                LocalCorner[1]], dtype = float)
			M_hat = np.dot(np.dot(K, Rt),M)
			xy = np.dot(Rt,M)
			M_hat = M_hat/M_hat[2,0]
			xy = xy/xy[2,0]
			u, v = M_hat[0,0], M_hat[1,0]
			x, y = xy[0,0], xy[1,0]
			m_hat = np.array([u+(u-cx)*(k1*(x**2+y**2)+k2*((x**2+y**2)**2)), v+(v-cx)*(k1*(x**2+y**2)+k2*((x**2+y**2)**2))], dtype=float)
			loss = np.linalg.norm((m-m_hat))
			Loss += loss
	Loss = Loss/(13*54) #loss per img
	return Loss

def LossToMinimize(params, H_list, LocalCorners, GlobalCorners):
	fx, fy, cx, cy, k1, k2 = params
	Transformation_Matrices = HomogenousMatrix(np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float32), H_list)
	Loss = LossFunction(fx, fy, cx, cy, k1, k2, Transformation_Matrices, LocalCorners, GlobalCorners)
	return Loss

def DrawCorners(images, fx, fy, cx, cy, k1, k2, Transformation_Matrices, GlobalCorners, result_path):
	K = [[fx, 0, cx],[0, fy, cy],[0, 0, 1]]
	K = np.array(K, np.float32)
	for i, img in enumerate(images):
		img = copy.deepcopy(img)
		img = cv2.undistort(img, K, np.array([k1, k2, 0, 0, 0], dtype = float))
		Rt = Transformation_Matrices[i]
		for j in range(GlobalCorners.shape[0]):
			GlobalCorner = GlobalCorners[j]
			M = [GlobalCorner[0], GlobalCorner[1], 1]
			M = np.array(M)
			M = M.reshape(3,1)
			M_hat = np.dot(np.dot(K, Rt),M)
			xy = np.dot(Rt,M)
			M_hat = M_hat/M_hat[2,0]
			xy = xy/xy[2,0]
			u, v = M_hat[0,0], M_hat[1,0]
			x, y = xy[0,0], xy[1,0]
			m_hat = np.array([u+(u-cx)*(k1*(x**2+y**2)+k2*((x**2+y**2)**2)), v+(v-cx)*(k1*(x**2+y**2)+k2*((x**2+y**2)**2))], dtype=float)
			cv2.circle(img, (int(m_hat[0]),int(m_hat[1])), 7, (0,0,255), -1)
		cv2.imwrite(result_path + "\\" + f"PredCorners{i}.png", img)

def main():
	basePath = dirname(abspath(__file__))
	image_path = basePath + f"\\Calibration_Imgs"
	# C:\Computer_vision\vbodda_hw1\results\OrigCorners
	result_pathCC = basePath + "\\results\\OrigCorners"
	result_pathPC = basePath + "\\results\\PredCorners"
	
	images = LoadImagesFromFolder(image_path)
	dims = (9,6)
	length = 21.5
	CornersLocal = ChessCornersLocal(images, dims, result_pathCC)
	#corners[imagenum][cornernm][0->x 1->y]
	CornersGlobal = ChessCornersGlobal(dims, length)
	H_list = Homography(CornersLocal, CornersGlobal)
	b = ClosedFormSolution(H_list)
	fx, fy, cx, cy = IntrinsicParams(b)
	k = [0, 0]
	K = [[fx, 0, cx],[0, fy, cy],[0, 0, 1]]
	K = np.array(K, np.float32)
	print(f"Parameters before optimization: {K}")
	print(f"Distortion Parameters before Optimization: {k[0]}, {k[1]}")
	Transformation_Matrices = HomogenousMatrix(K, H_list)
	LossPreOptimization = LossFunction(fx, fy, cx, cy, k[0], k[1], Transformation_Matrices, CornersLocal, CornersGlobal)
	print(f"Loss per corner before optimization: {LossPreOptimization}")
	params = [fx, fy, cx, cy, k[0], k[1]]
	print("---Optimizing---")
	resultant_parameters = optimize.least_squares(LossToMinimize, x0=params, method="trf", args=[H_list, CornersLocal, CornersGlobal])
	result = resultant_parameters.x
	fx, fy, cx, cy, k1, k2 = result
	KOptimized = np.array([[fx, 0, cx],[0, fy, cy],[0, 0, 1]])
	Radial_Distortion = [k1, k2]
	print(f"Parameters After Optimization: \n{KOptimized}")
	print(f"Radial Distortion After Optimization: {Radial_Distortion}")
	TransOptimized = HomogenousMatrix(KOptimized, H_list)
	LossOptimized = LossFunction(fx, fy, cx, cy, k1, k2,TransOptimized, CornersLocal, CornersGlobal)
	print(f"Optimised co-ordinates loss is: {LossOptimized}")
	DrawCorners(images, fx, fy, cx, cy, k1, k2, TransOptimized, CornersGlobal, result_pathPC)
	
if __name__ == "__main__":
	main()