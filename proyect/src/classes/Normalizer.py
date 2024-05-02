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
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 2))
            self.images[i] = clahe.apply(self.images[i])