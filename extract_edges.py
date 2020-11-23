# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 14:19:31 2020

@author: junea
"""

import numpy as np
import cv2
import matplotlib.pyplot as plot

# Import input image
image = plot.imread("./images/angio3superf.tif")
plot.subplot(1,2,1)
plot.axis("off")
plot.title("Input image")  
plot.imshow(image)
i_gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
size=i_gray.shape
                    
#Media de los píxeles
def media(size):
	sume = 0
	dim = size[0]*size[1]
	for i in range(size[0]):
		for j in range(size[1]):
			sume += i_gray[i,j]
	return sume/dim

media = media(size)

#Image processing
# 1. White top hat 
# Getting the kernel to be used in Top-Hat 
filterSize =(11,11) 
"MORPH_RECT / MORPH_CROSS / MORPH_ELLIPSE"
elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, filterSize) 

def whitetophat(image, elem):
    """The white_tophat of an image is defined as the image minus its 
       morphological opening. This operation returns the bright spots 
       of the image that are smaller than the structuring element"""
    
    image = image.copy()
    w_tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT,elem)
    return w_tophat

w_tophat = whitetophat(image, elem)
plot.subplot(1,2,2)
plot.axis("off")
plot.title("White top hat") 
plot.imshow(w_tophat)
 
# 2. Edges identification
def canny(w_tophat, filterSize, size):
    "Reducción de ruido mediante filtro Gaussiano"
    i_gaussian = cv2.GaussianBlur(w_tophat,filterSize,2)
    "Normalización de la imagen"
    i_normalized = cv2.normalize (i_gaussian,None,0,255,cv2.NORM_MINMAX)
    "Detección de bordes"
    i_edges = cv2.Canny (np.uint8(i_normalized),0, media+media*0.7*0.2)
    #edges = cv2.Canny(np.uint8(edges),t1,t2)     t2 = m+m*0.2*precision,2 ¿xq es este el máximo?
    return i_edges

i_edges = canny(w_tophat, filterSize, size)
plot.figure()
plot.axis("off")
plot.title("Canny - Edges identification") 
plot.imshow(i_edges, cmap='gray')

# 3. Closed / Opened: para eliminar la mayoría de los falsos positivos
def closed (i_edges, elem):
    i_closed = cv2.morphologyEx(i_edges,cv2.MORPH_CLOSE,elem)
    return i_closed

i_closed = closed(i_edges, elem)
plot.figure()
plot.subplot(1,2,1)
plot.axis("off")
plot.title("Closing") 
plot.imshow(i_closed, cmap='gray')

# Hacemos el negativo de la imagen 
i_inverted = cv2.bitwise_not(i_closed)

elem1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(13, 13)) 

def opened (i_inverted, elem):
    i_opened = cv2.morphologyEx(i_inverted,cv2.MORPH_OPEN,elem1)
    return i_opened

i_opened = opened(i_inverted, elem)

plot.subplot(1,2,2)
plot.axis("off")
plot.title("Opening") 
plot.imshow(i_opened, cmap='gray')

# 4. Erode: Ajustar con mayor precisión el contorno de la segmentación a los bordes vasculares circundantes (con las operaciones morfológicas previas pierden precisión)

elem1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1,1)) 

def erode (i_opened, elem1):
    i_eroded = cv2.erode(i_opened, elem1)
    return i_eroded

i_eroded = erode(i_opened, elem1)

plot.figure()
plot.axis("off")
plot.title("Eroded") 
plot.imshow(i_eroded, cmap='gray')

# 5. Modelo de contorno: eliminar falsos negativos

"Threshold"
ret,thresh = cv2.threshold(i_eroded, 217, 255,0)
contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) 

"Extraer columna de size"
i = 0
total_contours = np.ones(len(contours))
for i in range (len(contours)):
    contornos = contours[i].shape
    total_contours[i] = np.array(contornos[0])

"Elegir el contorno más grande"
max_contours = int(np.max(total_contours))
    
"Dibujar el contorno más grandes"
i = 0
for i in range (len(total_contours)):
    if total_contours[i] == max_contours:
       i_contours = cv2.drawContours(w_tophat, contours, i, (0, 255, 0), 1)
 
plot.figure()
plot.axis("off")
plot.title("Contour")
plot.imshow(i_contours)

# 6. Proceso de crecimiento de la región: para agregar o eliminar progresivamente los píxeles vecinos por similitud de intensidad hasta alcanzar todo el contorno del borde vascular

