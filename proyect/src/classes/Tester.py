
from classes.Detector import Detector
from typing import List
from cv2.typing import MatLike,Rect
import cv2
from common import FileFuncs as ff
from classes.Detector import Detector
from settings import IMAGES_PATH

class Tester:
    
    def __init__(self) -> None:
        pass
        
    def exec_general_test(self,startIndex:int=0,endIndex:int=101) -> None:
        #Borrar el contenido de los directorios:
        ff.remove_directory_content()
        ff.remove_directory_content(path=IMAGES_PATH+"cropped/")
        ff.remove_directory_content(path=IMAGES_PATH+"cropped_mask/")
        ff.remove_directory_content(path=IMAGES_PATH+"final_regioned/")
        
        #Leer las imagenes:
        images: List[MatLike] = ff.read_images(start=startIndex, end=endIndex)
        images_final_regioned: List[MatLike] = [img.copy() for img in images]
        
        #Crear el detector y mejorar el contraste:
        det: Detector = Detector(images)
        det.improve_contrast()
        
        #Obtener, filtrar y dibujar las regiones en una copia:
        filter_regions: List[tuple[List[Rect],int]] = det.filter_images_regions
        img_draws: List[MatLike] = det.draw_regions(filter_regions)
        ff.save_images(img_draws,cv2Const=cv2.COLOR_BGR2RGB)
        
        
        #Recortar las regiones y guardarlas:
        cropped_images: List[tuple[List[tuple[MatLike,Rect]],int]] = det.crop_regions(filter_regions)
        for crops,idx in cropped_images:
            imgs: List[MatLike] = [img for img,_ in crops]
            ff.save_images(imgs,cv2Const=cv2.COLOR_BGR2RGB,extra=f"{idx}-",path=IMAGES_PATH+"cropped/")
            
        #Aplicamos mascara:
        cropped_mask_images: List[tuple[List[tuple[MatLike,Rect]],int]] = det.apply_filter_cropped(cropped_images)
        for crops_mask,idx in cropped_mask_images:
            imgs = [img for img,_ in crops_mask]
            ff.save_images(imgs,extra=f"{idx}-",path=IMAGES_PATH+"cropped_mask/")
        
        det.draw_final_regions(images_final_regioned,cropped_mask_images)
        ff.save_images(images_final_regioned,cv2Const=cv2.COLOR_BGR2RGB,path=IMAGES_PATH+"final_regioned/")