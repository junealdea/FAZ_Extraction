# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 14:12:35 2020

@author: june.aldea
"""

import cv2
import numpy as np

input_image = "./images/angio3superf.tif"

# read image 
image = cv2.imread(input_image,0)           #plot.imshow(image, cmap='gray')

#Valor medio pixeles
def media(image):
	size = image.shape
	sume = 0
	dim = size[0]*size[1]                    #columna x fila
	for i in range(size[0]):
		for j in range(size[1]):
			sume += image[i,j]
	return sume/dim

valormedio = media(image)

#Datos imagenes de entrada
precision = 0.7                              #Valor entre 0-1
gamma = 2
t1 = 0
t2 = valormedio+valormedio*0.2*precision        
mm = 3                                       #Milímetros de la imagen (3 0 6)
prof = 0                                     #Profundidad de la imagen (0 o 1)

#Identificación de bordes
def canny(image,t1,t2,gamma):
	"""
	Implementación de un filtro Gaussiano para la reducción del ruido
    Estiramiento del histograma para el aumento del contraste
    Detección de bordes

	Entradas:
		- image: OCT-A image
		- t1 and t2:  thresholds of canny edge detector
		- gamma: gamma parameter to canny edge detector
	"""
    #Imagen de salida = cv2.GaussianBlur (Imagen de entrada, tamaño del kernel [anchura, altura], desviación estándar 
	filteredimage = cv2.GaussianBlur(image,(15,15),gamma)
    #Imagen de salida = cv2.normalize (Imagen de entrada, Imagen de salida, rango intensidad deseado (0,255), cv2.NORM_MINMAX)
	normalizedimage = cv2.normalize(filteredimage,0,255,cv2.NORM_MINMAX)
    #Imagen de salida = cv2.Canny (np.uint8(Imagen de entrada), t1, t2)
	edges = cv2.Canny(np.uint8(normalizedimage),t1,t2)

	return edges

Bordes = canny(image,t1,t2,gamma)

size = image.shape
kernelsize = max(((size[0]/100)-3),3)
kernelsize = kernelsize*2

#Operaciones morfológiccas

def morph(op,image, kernelsize):
    
	if (op == 'closed'):
		se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernelsize,kernelsize))
		imClosed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, se)
		return imClosed
	elif (op == 'open'):
		se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernelsize,kernelsize))
		imOpen = cv2.morphologyEx(image, cv2.MORPH_OPEN, se)
		return imOpen
	elif (op == 'tophat'):
		se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernelsize,kernelsize))
		th = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, se)
		return th
	elif (op == 'dilate'):
		se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernelsize,kernelsize))
		imDil = cv2.dilate((image *1.0).astype(np.float32),se)
		return imDil
	elif (op == 'erode'):
		se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernelsize,kernelsize))
		imEr = cv2.erode((image *1.0).astype(np.float32),se)
		return imEr

#Extraccíon de bordes
def edges_extraction(im,prof,mm,precision):
"
	size = im.shape
	if prof:
		# canny
		m = media(im)
		m = m*255
		edges = canny(im,0,m+m*0.2*precision,2)
		# closed
		ss = max(((size[0]/100)-3),3)
		ss = ss*2
		imClosed = morph('closed',edges,ss)

	else:
		# canny
		m = media(im)
		m = m*255
		if mm == 6:
			edges = canny(im,0,m+m*0.8*precision,1.9)
		edges = canny(im,0,m+m*0.3*precision,1.9)
		# closed
		ss = (size[0]/100)-3
		if ss<3:
			ss =3
		ss = ss*2
		imClosed = morph('closed',edges,ss)
	im = imClosed.copy()
	return edges, im
