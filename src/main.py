from fileinput import filename
import cv2 as cv
from cv2 import blur
import numpy as np

import sys

scale = 1
delta = 0
ddepth = cv.CV_16S

#  pegando o path da imagem para ler com o imread
path= r'C:\Users\Caio Peixoto\Documents\GitHub\PDI\Dataset\1.jpg'

# lendo a imagem
imgRGB = cv.imread(path)

imgRGBs = cv.resize(imgRGB, (940, 540))

# Alplicando borramento Gaussiano para eliminar ruídos
img_borrada= cv.GaussianBlur(imgRGBs,(5,5),0)

# convertendo de rgb para hsv
imgHSV = cv.cvtColor(img_borrada, cv.COLOR_RGB2HSV)

# Valor do Limite inferior para cor vermelha
lower1 = np.array([0, 100, 20])
upper1 = np.array([10, 255, 255])

# Valor do limite superior para cor vermelha
lower2 = np.array([115, 100, 20]) 
upper2 = np.array([179, 255, 255])
 
# Aplicando mascara na imagem com a função cv.inRange() 
# que é semelhante a bwareaopen() utilizada no artigo

lower_mask = cv.inRange(imgHSV, lower1, upper1)
upper_mask = cv.inRange(imgHSV, lower2, upper2) 

mask = lower_mask + upper_mask

# Encontrar os contornos 
contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# Criar uma máscara em branco do mesmo tamanho da imagem
mask_preenchida = np.zeros_like(mask)

# Preencher os contornos na máscara
cv.drawContours(mask_preenchida, contours, -1, 255, thickness=cv.FILLED)


grad_x = cv.Sobel(mask_preenchida, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)

# Gradient-Y
# grad_y = cv.Scharr(gray,ddepth,0,1)
grad_y = cv.Sobel(mask_preenchida, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)


abs_grad_x = cv.convertScaleAbs(grad_x)
abs_grad_y = cv.convertScaleAbs(grad_y)

# grad vai mostrar somente as bordas da mascara
grad = cv.addWeighted(abs_grad_x, 1, abs_grad_y, 1, 0)

######################################################################################################################
# Tentar entender melhor como aplicar a tranformada de hough
# Definir o kernel para a erosão
kernel = np.ones((3, 3), np.uint8)  # Exemplo de um kernel 3x3

# Aplicar a erosão
eroded_image = cv.erode(mask_preenchida, kernel, iterations=1)

# discover image contours
contours, _= cv.findContours(image=eroded_image, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_NONE)

edges = cv.Canny(eroded_image, 50, 150)

# Aplicar a Transformada de Hough para detectar linhas
lines = cv.HoughLines(edges, 1, np.pi/180, 200)

# Ordenar as linhas com base no número de votos (votos em ordem decrescente)
if lines is not None:
    lines = sorted(lines, key=lambda x: -x[0][0])

    # Selecionar as 8 melhores linhas
    selected_lines = lines[:8]

    # Desenhar as linhas selecionadas na imagem original
    for line in selected_lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv.line(imgRGB, (x1, y1), (x2, y2), (0, 0, 255), 2)
######################################################################################################################
# Exibir a imagem original com as linhas detectadas
cv.imshow('Detected Lines', imgRGB)

# Exibir imagem erodida
# cv.imshow('Eroded Image', eroded_image)

# Exibir a imagem resultante
# cv.imshow('Imagem Preenchida', mask_preenchida)
# cv.imshow('Contornos', grad)

# # mostrando imagem
# cv.imshow('Imagem Original',imgRGB)
# cv.imshow('Imagem Gaussiano', img_borrada)
# cv.imshow('full-mask',mask)

cv.waitKey(0)
cv.destroyAllWindows