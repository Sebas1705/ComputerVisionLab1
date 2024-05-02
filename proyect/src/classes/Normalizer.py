from typing import List
from cv2.typing import Rect
import cv2
from cv2.typing import MatLike
import numpy as np

class Normalizer:
    
    def __init__(self,images: List[MatLike]) -> None:
        self.images: List[MatLike] = images
        self.__nImages: int = len(images)
        
        
    def clahe_apply(
        self
    ) -> None:
        for i in range(self.__nImages):
            self.images[i] = cv2.cvtColor(self.images[i], cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            self.images[i] = clahe.apply(self.images[i])
            
    def equalize(
        self
    ) -> None:
        for i in range(self.__nImages):
            self.images[i] = cv2.cvtColor(self.images[i], cv2.COLOR_BGR2GRAY)
            self.images[i] = cv2.equalizeHist(self.images[i])
            self.images[i] = cv2.cvtColor(self.images[i], cv2.COLOR_GRAY2BGR)
            
    def umbralize(
        self
    ) -> None:
        for i in range(self.__nImages):
            self.images[i] = cv2.cvtColor(self.images[i], cv2.COLOR_BGR2GRAY)
            self.images[i] = cv2.adaptiveThreshold(self.images[i], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
