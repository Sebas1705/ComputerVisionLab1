from typing import List
from cv2.typing import Rect
import cv2
from cv2.typing import MatLike
import common.FileFuncs as ff
import numpy as np
from settings import IMAGES_PATH

class Normalizer:
    
    def __init__(self,images: List[MatLike]) -> None:
        self.images: List[MatLike] = images
        self.__nImages: int = len(images)
        
        
    def clahe_apply(
        self
    ) -> None:
        for i in range(self.__nImages):
            self.images[i] = cv2.cvtColor(self.images[i], cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 2))
            self.images[i] = clahe.apply(self.images[i])
            
    def corregir_perspectiva(
        self
    ) -> None:
        ind = 0
        for i in range(self.__nImages):
            # Aplicar filtro de Canny para detectar bordes
            bordes = cv2.Canny(self.images[i], 50, 150, apertureSize=3)
            # Detectar líneas utilizando la transformada de Hough
            lineas = cv2.HoughLinesP(bordes, 1, np.pi/180, threshold=100, minLineLength=40, maxLineGap=10)
            if lineas is not None:
                # Calcular el ángulo promedio de las líneas detectadas
                angulo_promedio = np.mean([np.arctan2(y2 - y1, x2 - x1) for line in lineas for x1, y1, x2, y2 in line])
                # Rotar la imagen según el ángulo promedio
                (alto, ancho) = self.images[i].shape[:2]
                centro = (ancho // 2, alto // 2)
                matriz_rotacion = cv2.getRotationMatrix2D(centro, np.degrees(angulo_promedio), 1.0)
                self.images[i] = cv2.warpAffine(self.images[i], matriz_rotacion, (ancho, alto), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                print(f"Not none: ${i}")
            else:
                ind+=1
        print("Nones: "+str(ind))