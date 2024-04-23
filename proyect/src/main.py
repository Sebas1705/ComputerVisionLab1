
from typing import List
from cv2.typing import MatLike
import cv2
from common import FileFuncs as ff
from classes.Detector import Detector
from settings import GLOBAL_PATH


if __name__ == "__main__":
    #Borrar el contenido de los directorios:
    ff.remove_directory_content()
    ff.remove_directory_content(path=f"{GLOBAL_PATH}/proyect/images/cropped/")
    ff.remove_directory_content(path=f"{GLOBAL_PATH}/proyect/images/cropped_mask/")
    
    #Leer las imagenes:
    images: List[MatLike] = ff.read_images(end=10)
    images2: List[MatLike] = ff.read_images(end=10)
    
    #Crear el detector y mejorar el contraste:
    det: Detector = Detector(images)
    det.improve_contrast()
    
    #Obtener, filtrar y dibujar las regiones en una copia:
    filter_regions = det.filter_images_regions
    img_draws: List[MatLike] = det.draw_regions(filter_regions)
    ff.save_images(img_draws,cv2Const=cv2.COLOR_BGR2RGB)
    
    
    #Recortar las regiones y guardarlas:
    cropped_images = det.crop_regions(filter_regions)
    for crops,idx,region in cropped_images:
        ff.save_images(crops,cv2Const=cv2.COLOR_BGR2RGB,extra=f"{idx}-",path=f"{GLOBAL_PATH}/proyect/images/cropped")
        
    #Aplicamos mascara:
    cropped_mask_images = det.apply_filter_cropped(cropped_images)
    for crops_mask,idx in cropped_mask_images:
        ff.save_images(crops_mask,cv2Const=None,extra=f"{idx}-",path=f"{GLOBAL_PATH}/proyect/images/cropped_mask")
    
    det.draw_final_regions(images2,cropped_mask_images)
    ff.save_images(images2,cv2Const=None,extra=f"{idx}-",path=f"{GLOBAL_PATH}/proyect/images/test")