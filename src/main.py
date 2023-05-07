from fileinput import filename
import cv2 as cv
from cv2 import blur
import numpy as np

import sys

#  pegando o path da imagem para ler com o imread
path= r'C:\Users\Caio Peixoto\Documents\GitHub\PDI\Dataset\1.jpg'

# lendo a imagem
imgRGB = cv.imread(path)

# convertendo de rgb para hsv
imgHSV = cv.cvtColor(imgRGB, cv.COLOR_RGB2HSV)

# Valor do Limite inferior para cor vermelha
lower1 = np.array([0, 100, 20])
upper1 = np.array([10, 255, 255])

# Valor do limite superior para cor vermelha
lower2 = np.array([115, 100, 20]) 
upper2 = np.array([179, 255, 255])
 
# Aplicando mascara na imagem
lower_mask = cv.inRange(imgHSV, lower1, upper1)
upper_mask = cv.inRange(imgHSV, lower2, upper2) 

mask = lower_mask + upper_mask



# mostrando imagem
cv.imshow('Imagem Original',imgRGB)
cv.imshow('full-mask',mask)

cv.waitKey(0)
cv.destroyAllWindows