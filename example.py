# -*- encoding: utf-8 -*-
import cv2
import opsfaz as faz
import numpy as np
import drawfaz as draw

# Import input image
input_image = ("./data/A01_L/PARon.png")
imagetocrop = cv2.imread(input_image,0)
# Cortamos la imagen
size = imagetocrop.shape
image = imagetocrop[150:640, 130:600]

# configure parameters
mm = 3
deep = 0
precision = 0

# call the function
faz, area, cnt = faz.detectFAZ(image, mm, deep, precision) 
#	Outputs:
#	- faz is a binary image with the region of the FAZ as mask
#	- area is the area of the FAZ
#	- cnt is the contour in opencv that represents the contour of the FAZ



# we obtain the faz mask
cv2.imshow("Cropped image",image)
cv2.imshow("Mask extracted",faz)
cv2.waitKey(0)
cv2.destroyAllWindows()


print ("Area:	"+ str(area)+" square mm")