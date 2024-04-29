from functools import lru_cache
from typing import List, Sequence
from cv2.typing import Rect
import cv2
from cv2.typing import MatLike
import numpy as np
from settings import *
import common.FileFuncs as ff

class Detector:
    
    lower_blue = np.array([100,50,60])
    upper_blue = np.array([130,255,255])

    def __init__(
        self,
        images:List[MatLike]
    ) -> None:
        """
        Initialize the instance of the Detector class.
        
        Parameters:
        -----------
        images : List[MatLike]
            The list of input images.
        """
        self.__images: List[MatLike] = images  
        self.gray_images: List[MatLike] = [cv2.cvtColor(img,cv2.COLOR_RGB2GRAY) for img in self.__images]
        self.__nImages: int = len(self.__images)
        self.__mser: cv2.MSER = cv2.MSER_create(
            delta=DELTA, 
            min_area=MIN_AREA, 
            max_area=MAX_AREA, 
            max_variation=MAX_VARIATION, 
            min_diversity=MIN_DIVERSITY
        )
    
    def improve_contrast(
        self
    ) -> None:
        """
        Adjusts the contrast of each image in the dataset using the histogram equalize.
        
        Parameters:
        -----------
        self : Detector
            The instance of the Detector class.
        """
        for i in range(self.__nImages):
            self.gray_images[i] = cv2.equalizeHist(self.gray_images[i])

    @property
    @lru_cache
    def list_images_regions(
        self
    ) -> list[tuple[List[Rect],int]]:
        """
        Returns a list of tuples, where each tuple contains a list of rectangles and the index of the image they belong to.

        The rectangles are obtained by running the MSER algorithm on each image, and then grouping the resulting rectangles using the cv2.groupRectangles function.

        Parameters:
        -----------
        self : Detector
            The instance of the Detector class.

        Returns:
        --------
        list[tuple[List[Rect], int]]
            A list of tuples, where each tuple contains a list of rectangles and the index of the image they belong to.
        """
        return [
            (self.__mser.detectRegions(self.gray_images[idx])[1],idx) #Create a sequence of rectangles
            for idx in range(self.__nImages)
        ]
        
    @property
    @lru_cache
    def groupped_images_regions(
        self
    ) -> List[tuple[List[Rect],int]]:
        """
        Returns a list of tuples, where each tuple contains a list of rectangles and the index of the image they belong to.

        The rectangles are obtained by running the MSER algorithm on each image, and then grouping the resulting rectangles using the cv2.groupRectangles function.

        Parameters:
        -----------
        self : Detector
            The instance of the Detector class.

        Returns:
        --------
        list[tuple[List[Rect], int]]
            A list of tuples, where each tuple contains a list of rectangles and the index of the image they belong to.
        """
        return [
            #Group a sequence of rectangles
            (cv2.groupRectangles(rects[0], GROUP_THRESHOLD, EPS)[0],rects[1]) 
            for rects in self.list_images_regions
        ]
        
    @property
    @lru_cache
    def filter_images_regions(
        self
    ) -> List[tuple[List[Rect],int]]:
        """
        Returns a list of tuples, where each pass a filter on the rectangles.
        
        Parameters:
        -----------
        self : Detector
            The instance of the Detector class.

        Returns:
        --------
        list[tuple[List[Rect], int]]
            A list of tuples, where each tuple contains a list of rectangles and the index of the image they belong to.
        """
        filter_images: List[tuple[List[Rect],int]] = []
        for groupped in self.groupped_images_regions:
            filter_rects: List[Rect] = []
            for rect in groupped[0]:
                x, y, w, h = rect
                aspect_ratio = float(w) / h
                if MIN_RATIO < aspect_ratio < MAX_RATIO:
                    if w <= MAX_WIDTH and h <= MAX_HEIGHT:                        
                            #Enlarge the region
                            max_image_width = self.__images[groupped[1]].shape[1]
                            max_image_height = self.__images[groupped[1]].shape[0]
                            x=x-ENLARGE_WIDTH if x-ENLARGE_WIDTH>=0 else 0
                            y=y-ENLARGE_HEIGHT if y-ENLARGE_HEIGHT>=0 else 0
                            w=w+ENLARGE_WIDTH*2 if w+ENLARGE_WIDTH*2<=max_image_width else max_image_width
                            h=h+ENLARGE_HEIGHT*2 if h+ENLARGE_HEIGHT*2<=max_image_height else max_image_height
                            rect: Rect = x,y,w,h
                            filter_rects.append(rect)
            filter_images.append((filter_rects,groupped[1]))
        return filter_images
    
    def draw_regions(
        self,
        regions:List[tuple[List[Rect],int]]
    ) -> List[MatLike]:
        """
        Draws bounding boxes on the images in the dataset.

        Parameters:
        -----------
        self : Detector
            The instance of the Detector class.
        regions : List[tuple[List[Rect],int]]
            A tuple list of regions with it's image index

        Returns:
        --------
        List[MatLike]
            A list of the images with bounding boxes drawn on them.
        """
        images_copy: List[MatLike] = []
        for img in self.__images:
            images_copy.append(img.copy())
        for regions,idx in regions:
            for x,y,w,h in regions:
                cv2.rectangle(images_copy[idx],(x,y),(x+w, y+h),COLOR_BORDER,THICKNESS)
        return images_copy
    
    def crop_regions(
        self,
        regions: List[tuple[List[Rect],int]]
    ) -> List[tuple[List[tuple[MatLike,Rect]],int]]:
        """
        Crops the regions of interest from the input images.

        Parameters:
        -----------
        self : Detector
            The instance of the Detector class.
        regions : List[tuple[List[Rect],int]]
            A tuple list of regions with it's image index

        Returns:
        --------
        List[tuple[List[tuple[MatLike,Rect]],int]]
            A list of tuples, where each tuple contains a list of cropped images and its regions and the index of the image it belongs to.
        """
        cropped_images = []
        for regions_c,idx in regions:
            cropped_index_images = []            
            for region in regions_c:
                x, y, w, h = region  
                cropped_image: MatLike = cv2.resize(self.__images[idx][y:y+h,x:x+w],CROPPED_TAM)
                cropped_index_images.append((cropped_image,region))
            cropped_images.append((cropped_index_images,idx))
        return cropped_images
        
    def apply_filter_cropped(
        self,
        cropped_images:List[tuple[List[tuple[MatLike,Rect]],int]]
    )-> List[tuple[List[tuple[MatLike,Rect]],int]]:
        """
        Filter cropped images by a mask, and it percentage of relative non black pixels.

        Parameters:
        -----------
        self : Detector
            The instance of the Detector class.
        cropped_images : List[tuple[List[tuple[MatLike,Rect]],int]]
            A list of tuples, where each tuple contains a list of cropped images and its regions and the index of the image it belongs to.

        Returns:
        --------
        List[tuple[List[tuple[MatLike,Rect]],int]]
            A list of tuples, where each tuple contains a list of cropped filter images and its regions and the index of the image it belongs to.
        """
        cropped_mask:List[tuple[List[tuple[MatLike,Rect]],int]] = []
        for croppeds,idx in cropped_images:
            croppeds_image: List[tuple[MatLike,Rect]]=[]
            for cropped,region in croppeds:
                hsv_cropped = cv2.cvtColor(cropped,cv2.COLOR_RGB2HSV)
                mask = cv2.inRange(hsv_cropped,self.lower_blue,self.upper_blue)
                bitwise = cv2.cvtColor(cv2.cvtColor(cv2.bitwise_and(hsv_cropped,hsv_cropped,mask=mask),cv2.COLOR_HSV2BGR),cv2.COLOR_BGR2RGB)
                gris = cv2.cvtColor(bitwise, cv2.COLOR_BGR2GRAY)
                # Contar los píxeles negros (píxeles con valor 0)
                total_pixeles = gris.shape[0] * gris.shape[1]
                pxeles_negros = total_pixeles - cv2.countNonZero(gris)
                # Calcular el porcentaje de píxeles negros
                porcentaje = (pxeles_negros / total_pixeles) * 100
                if porcentaje < 13:
                    croppeds_image.append((bitwise,region))
            cropped_mask.append((croppeds_image,idx))
        return cropped_mask
    
    def draw_final_regions(
        self,
        dest_images:List[MatLike],
        cropped_filter_images:List[tuple[List[tuple[MatLike,Rect]],int]]
    ) -> str:
        string = ''
        for tuple,idx in cropped_filter_images:
            for _,reg in tuple:
                x,y,w,h=reg
                cv2.rectangle(dest_images[idx],(x,y),(x+w,y+h),COLOR_BORDER,THICKNESS)
                string = string + str(idx) + ';' + str(x) + ';' + str(y) + ';' + str(x+w) + ';' + str(y+h)  + '\n'                
        return string