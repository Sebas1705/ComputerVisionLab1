import cv2
import numpy as np

def corregir_perspectiva(imagen):
    # Convertir la imagen a escala de grises
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    # Aplicar filtro de Canny para detectar bordes
    bordes = cv2.Canny(gris, 50, 150, apertureSize=3)
    
    # Detectar líneas utilizando la transformada de Hough
    lineas = cv2.HoughLinesP(bordes, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
    
    # Calcular el ángulo promedio de las líneas detectadas
    angulo_promedio = np.mean([np.arctan2(y2 - y1, x2 - x1) for line in lineas for x1, y1, x2, y2 in line])
    
    # Rotar la imagen según el ángulo promedio
    (alto, ancho) = imagen.shape[:2]
    centro = (ancho // 2, alto // 2)
    matriz_rotacion = cv2.getRotationMatrix2D(centro, np.degrees(-angulo_promedio), 1.0)
    imagen_rotada = cv2.warpAffine(imagen, matriz_rotacion, (ancho, alto), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    return imagen_rotada

# Cargar la imagen
imagen = cv2.imread('./proyecto/images/i_final_cropped/8-00000.png')


# Corregir la perspectiva
imagen_corregida = corregir_perspectiva(imagen)

# Mostrar la imagen original y la imagen corregida
cv2.imshow('Imagen Original', imagen)
cv2.imshow('Imagen Corregida', imagen_corregida)
cv2.waitKey(0)
cv2.destroyAllWindows()
