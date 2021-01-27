# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 13:49:02 2021

@author: june.aldea
"""

import pandas as pd
import os
import cv2
import re

#Extracción de imágenes
directory = r'C:\Users\june.aldea\Dropbox\TFG_June_Aldea\Codigo\FAZ_Extraction_june\images'

#Extraccion de datos del excel
data_list = pd.read_excel('angioct_list_2021.01.19.xlsx')
size_data = data_list.shape

#Image scale
roots = []
for root, dirnames, filenames in os.walk(directory):
    for filename in filenames:
        if filename.endswith(('.png')):
            roots.append(os.path.join(root, filename))  
size_roots = len(roots)

for subdir in roots:
    images = cv2.imread(subdir)
    for i in range(size_roots):
        name_split = re.split("[ ,_]", filenames[i])
        for i in range(size_data[0]-1):
            if data_list.ID[i] == name_split[1] and name_split[2] == "OD":
                scaleX = data_list.scaleX[i]
            else:
                scaleX = data_list.scaleX[i+1]
    
#CORREGIR LA ÚLTIMA PARTE DEL IF