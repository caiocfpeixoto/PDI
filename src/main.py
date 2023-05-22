import cv2 as cv
import numpy as np

def find_red_objects(image):
    # Convertendo a imagem para o espaço de cores HSV
    hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    # Definindo os limites inferior e superior para detecção de vermelho
    lower_red = np.array([0, 50, 50])  # Matiz mínimo, saturação mínima e valor mínimo
    upper_red = np.array([10, 255, 255])  # Matiz máximo, saturação máxima e valor máximo

    # Criando uma máscara para pixels na faixa de cores vermelhas
    mask = cv.inRange(hsv_image, lower_red, upper_red)

    # Aplicando a máscara à imagem original
    red_objects = cv.bitwise_and(image, image, mask=mask)

    # Aplicando a função bwareaopen para remover objetos pequenos
    gray_image = cv.cvtColor(red_objects, cv.COLOR_BGR2GRAY)
    _, threshold = cv.threshold(gray_image, 1, 255, cv.THRESH_BINARY)
    filtered_objects = cv.bitwise_and(red_objects, red_objects, mask=threshold)

    return filtered_objects

# Carregando a imagem
img = cv.imread(cv.samples.findFile("Dataset/nickr.jpg"))

# Resize in case of a image being too big.
imgRGB = cv.resize(img, (940, 540))

# Alplicando borramento Gaussiano para eliminar ruídos
img_borrada= cv.GaussianBlur(imgRGB,(5,5),0)

# Chamando a função para encontrar os objetos vermelhos
red_objects = find_red_objects(img_borrada)

def detect_edges(image):
    # Convertendo a imagem para tons de cinza
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Aplicando o filtro de Sobel para detecção de bordas
    sobel_x = cv.Sobel(gray_image, cv.CV_64F, 1, 0, ksize=3)
    sobel_y = cv.Sobel(gray_image, cv.CV_64F, 0, 1, ksize=3)

    # Calculando a magnitude dos gradientes
    gradient_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    # Normalizando a magnitude dos gradientes para o intervalo [0, 255]
    gradient_magnitude_normalized = cv.normalize(gradient_magnitude, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)

    # Aplicando uma limiarização para binarizar a imagem das bordas
    _, edges = cv.threshold(gradient_magnitude_normalized, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    return edges

# Chamando a função para detectar as bordas na imagem segmentada
edges = detect_edges(red_objects)

def dilate_edges(edges):
    # Definindo o kernel para a dilatação
    kernel = np.ones((3, 3), np.uint8)

    # Realizando a dilatação nas bordas
    dilated_edges = cv.dilate(edges, kernel, iterations=1)

    return dilated_edges

# Chamando a função para realizar a dilatação nas bordas
dilated_edges = dilate_edges(edges)

def fill_objects(edges):
    # Copiando a imagem das bordas para preservar a original
    filled_image = edges.copy()

    # Encontrando os contornos dos objetos nas bordas
    contours, _ = cv.findContours(filled_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Preenchendo os objetos conectados
    for contour in contours:
        cv.drawContours(filled_image, [contour], 0, (255), -1)

    return filled_image

# Chamando a função para preencher o interior dos objetos conectados
filled_image = fill_objects(dilated_edges)

def remove_border_objects(image):
    # Encontrando os contornos dos objetos na imagem
    contours, _ = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Encontrando as dimensões da imagem
    height, width = image.shape[:2]

    # Definindo uma região de interesse (ROI) com base nas bordas
    roi = np.zeros_like(image)
    cv.rectangle(roi, (0, 0), (width - 1, height - 1), 255, -1)

    # Verificando quais objetos estão em contato com a ROI
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        if cv.pointPolygonTest(contour, (x, y), False) == 1.0:
            cv.drawContours(image, [contour], 0, 0, -1)

    return image

# Chamando a função para remover objetos conectados em contato com as bordas
processed_image = remove_border_objects(filled_image)

def improve_object_definition(image):
    # Definindo o kernel para a erosão
    kernel = np.ones((3, 3), np.uint8)

    # Realizando a erosão na imagem
    eroded_image = cv.erode(image, kernel, iterations=1)

    return eroded_image

# Chamando a função para melhorar a definição do objeto detectado
improved_image = improve_object_definition(processed_image)

def detect_lines(image):
    # Aplicando o detector de bordas Canny
    edges = cv.Canny(image, 50, 150)
    
    # Aplicando a transformada de Hough para encontrar as retas principais
    lines = cv.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)
   
    # Selecionando as oito melhores linhas
    if lines is not None:
        lines = lines[:8]

    # Unindo linhas próximas
    merged_lines = merge_lines(lines, max_distance=5)
    
    return merged_lines

def merge_lines(lines, max_distance):
    merged_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]

        # Calculando o ponto médio da linha
        mid_point = ((x1 + x2) // 2, (y1 + y2) // 2)

        # Verificando se o ponto médio está próximo de alguma linha já existente
        merge_flag = False
        for merged_line in merged_lines:
            existing_mid_point = ((merged_line[0][0] + merged_line[1][0]) // 2, (merged_line[0][1] + merged_line[1][1]) // 2)

            # Verificando a distância entre os pontos médios
            distance = np.sqrt((mid_point[0] - existing_mid_point[0]) ** 2 + (mid_point[1] - existing_mid_point[1]) ** 2)
            if distance <= max_distance:
                # Atualizando a linha existente com os novos pontos
                merged_line[0] = (min(merged_line[0][0], x1), min(merged_line[0][1], y1))
                merged_line[1] = (max(merged_line[1][0], x2), max(merged_line[1][1], y2))
                merge_flag = True
                break

        # Se não houver linha próxima, adicionamos uma nova linha
        if not merge_flag:
            merged_lines.append([(x1, y1), (x2, y2)])

    return merged_lines

def draw_lines(image, lines):
    line_color = (0, 255, 255)  # Amarelo

    for line in lines:
        start_point = line[0]
        end_point = line[1]

        # Desenhando a linha
        cv.line(image, start_point, end_point, line_color, thickness=2)

        # Desenhando os pontos de início e fim da linha
        cv.circle(image, start_point, radius=3, color=(0, 0, 255), thickness=-1)  # Vermelho
        cv.circle(image, end_point, radius=3, color=(0, 0, 255), thickness=-1)  # Vermelho

    return image

def draw_bounding_rectangle(image, lines):
    top_left = (image.shape[1], image.shape[0])
    bottom_right = (0, 0)

    for line in lines:
        start_point = line[0]
        end_point = line[1]

        # Atualizando as coordenadas do retângulo delimitador
        top_left = (min(top_left[0], start_point[0]), min(top_left[1], start_point[1]))
        top_left = (min(top_left[0], end_point[0]), min(top_left[1], end_point[1]))
        bottom_right = (max(bottom_right[0], start_point[0]), max(bottom_right[1], start_point[1]))
        bottom_right = (max(bottom_right[0], end_point[0]), max(bottom_right[1], end_point[1]))

    # Desenhando o retângulo delimitador
    cv.rectangle(image, top_left, bottom_right, color=(0, 255, 0), thickness=2)

    return image

# Detectando as linhas na improved_image
lines = detect_lines(improved_image)

# Desenhando as linhas na imagem original
image_with_lines = draw_lines(imgRGB.copy(), lines)

# Desenhando o retângulo delimitador
image_with_rectangle = draw_bounding_rectangle(image_with_lines, lines)

# Exibindo a imagem com as retas e o retângulo delimitador
cv.imshow('Objetos Vermelhos', red_objects)
cv.imshow('Imagem Original com Retas e Retângulo', image_with_rectangle)
cv.imshow('Imagem com Definicao Melhorada', improved_image)
# cv.imshow('Imagem Processada', processed_image)
# cv.imshow('Objetos Preenchidos', filled_image)
# cv.imshow('Bordas', edges)
# cv.imshow('Bordas Dilatadas', dilated_edges)
cv.waitKey(0)
cv.destroyAllWindows()