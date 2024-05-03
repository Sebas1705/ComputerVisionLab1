from typing import List
from cv2.typing import Rect
import cv2
from cv2.typing import MatLike
import common.FileFuncs as ff
import numpy as np
from settings import *

class Normalizer:
    
    def __init__(self,images: List[MatLike]) -> None:
        self.images: List[MatLike] = images
        self.__nImages: int = len(images)
        
        
    def clahe_apply(
        self
    ) -> None:
        """
        Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to each image in the list.

        Parameters:
        self (Normalizer): The instance of the Normalizer class.

        Returns:
        None: The method modifies the images in-place.

        Note:
        This method converts each image to grayscale before applying the CLAHE.
        The CLIP_LIMIT and TITLE_GRID_SIZE constants are used as parameters for the cv2.createCLAHE function.
        """
        for i in range(self.__nImages):
            self.images[i] = cv2.cvtColor(self.images[i], cv2.COLOR_BGR2GRAY)
            clahe: cv2.CLAHE = cv2.createCLAHE(CLIP_LIMIT,TITLE_GRID_SIZE)
            self.images[i] = clahe.apply(self.images[i])
            
    def perspective_correct(
        self
    ) -> None:
        for i in range(self.__nImages):
            # Aplicar filtro de Canny para detectar bordes
            bordes = cv2.Canny(self.images[i],THREASHOLD1,THREASHOLD2,apertureSize=APERTURE_SIZE)
            # Detectar líneas utilizando la transformada de Hough
            lineas = cv2.HoughLinesP(bordes,RHO,THETA,THREASHOLD,minLineLength=MIN_LINE_LENGTH, maxLineGap=MAX_LINE_GAP)
            if lineas is not None:
                # Calcular el ángulo promedio de las líneas detectadas
                angulo_promedio = np.mean([np.arctan2(y2 - y1, x2 - x1) for line in lineas for x1, y1, x2, y2 in line])
                # Rotar la imagen según el ángulo promedio
                (alto, ancho) = self.images[i].shape[:2]
                centro = (ancho // 2, alto // 2)
                matriz_rotacion = cv2.getRotationMatrix2D(centro, np.degrees(angulo_promedio), 1.0)
                self.images[i] = cv2.warpAffine(self.images[i], matriz_rotacion, (ancho, alto), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)