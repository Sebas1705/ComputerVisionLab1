import cv2
import numpy as np

# Leer la imagen y convertirla a escala de grises
image = cv2.imread('./proyect/images/test/00003.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Crear el detector MSER
mser = cv2.MSER_create()

# Detectar regiones de alto contraste
regions, _ = mser.detectRegions(gray)

# Inicializar lista para almacenar los rectángulos de las regiones detectadas
rectangles = []

# Procesar cada región detectada
for region in regions:
    # Obtener las coordenadas del rectángulo que envuelve la región
    x, y, w, h = cv2.boundingRect(region)
    
    # Calcular la relación de aspecto
    aspect_ratio = w / h
    
    # Filtrar regiones con relación de aspecto adecuada
    if 0.5 <= aspect_ratio <= 2.0:
        # Agrandar el rectángulo para incluir el borde blanco
        x -= 10
        y -= 10
        w += 20
        h += 20
        
        rectangles.append((x, y, w, h))

# Mostrar las regiones detectadas
for (x, y, w, h) in rectangles:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow('MSER Regions', image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Convertir la imagen a espacio de color HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Definir el rango de color azul saturado en HSV
lower_blue = np.array([90, 50, 50])
upper_blue = np.array([130, 255, 255])

# Crear la máscara
mask = cv2.inRange(hsv, lower_blue, upper_blue)

# Aplicar la máscara a la imagen original
blue_pixels = cv2.bitwise_and(image, image, mask=mask)

cv2.imshow('Blue Pixels', blue_pixels)
cv2.waitKey(0)
cv2.destroyAllWindows()

# # Definir la máscara de color azul saturado ideal
# ideal_mask = np.ones((40, 80), dtype=np.uint8) * 255

# # Inicializar la lista para almacenar los scores de las ventanas detectadas
# scores = []

# # Procesar cada región detectada por MSER
# for (x, y, w, h) in rectangles:
#     # Recortar la región detectada y cambiar su tamaño
#     region_resized = cv2.resize(gray[y:y+h, x:x+w], (80, 40))
    
#     # Extraer la máscara de color azul saturado
#     blue_mask = cv2.inRange(cv2.cvtColor(cv2.resize(image[y:y+h, x:x+w], (80, 40)), cv2.COLOR_BGR2HSV), lower_blue, upper_blue)
    
#     # Correlar la máscara con la máscara ideal
#     correlation = cv2.matchTemplate(blue_mask, ideal_mask, cv2.TM_CCOEFF_NORMED)[0][0]
    
#     # Establecer el score como la correlación
#     scores.append(correlation)

# # Mostrar los scores
# print(scores)

# Definir la función para calcular el índice de solapamiento entre dos rectángulos
def overlap_ratio(rect1, rect2):
    x1 = max(rect1[0], rect2[0])
    y1 = max(rect1[1], rect2[1])
    x2 = min(rect1[0] + rect1[2], rect2[0] + rect2[2])
    y2 = min(rect1[1] + rect1[3], rect2[1] + rect2[3])
    
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    union_area = rect1[2] * rect1[3] + rect2[2] * rect2[3] - intersection_area
    
    overlap_ratio = intersection_area / union_area
    return overlap_ratio

# Definir un umbral de solapamiento
overlap_threshold = 0.5

# Inicializar lista para almacenar las detecciones filtradas
filtered_rectangles = []

# Iterar sobre todas las ventanas y seleccionar una única ventana por cada sub-panel
for i, rect1 in enumerate(rectangles):
    if i not in filtered_rectangles:
        for j, rect2 in enumerate(rectangles):
            if i != j and overlap_ratio(rect1, rect2) > overlap_threshold:
                if rect1[2] * rect1[3] > rect2[2] * rect2[3]:
                    filtered_rectangles.append(j)
                else:
                    filtered_rectangles.append(i)

# Eliminar las detecciones duplicadas
filtered_rectangles = list(set(range(len(rectangles))) - set(filtered_rectangles))

# Mostrar las detecciones filtradas
for idx in filtered_rectangles:
    rect = rectangles[idx]
    cv2.rectangle(image, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 2)

cv2.imshow('Filtered MSER Regions', image)
cv2.waitKey(0)
cv2.destroyAllWindows()