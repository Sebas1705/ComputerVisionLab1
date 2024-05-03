
from classes.Detector import Detector
from typing import List
from cv2.typing import MatLike,Rect
import cv2
from common import FileFuncs as ff
from classes.Detector import Detector
from classes.Normalizer import Normalizer
from settings import IMAGES_PATH
from settings import FILES_PATH

class Tester:
    
    def __init__(self) -> None:
        pass
        
    def exec_general_test(
        self,
        images:List[MatLike],
        nameFiles:list[str]
    ) -> List[tuple[List[MatLike],int]]:
        
        #Borrar el contenido de los directorios:
        ff.remove_images_dests()
        
        #Copiar las imagenes:
        images_final_regioned: List[MatLike] = [img.copy() for img in images]
        
        #Crear el detector y mejorar el contraste:
        det: Detector = Detector(images)
        
        #Mejorar las imagenes en grises
        ff.save_images(det.gray_images,path=IMAGES_PATH+"a_gray_before/")
        det.improve_contrast(det.gray_images)
        ff.save_images(det.gray_images,path=IMAGES_PATH+"b_gray_after/")
        
        #Recoger las regiones:
        regions = det.list_images_regions
        img_draws: List[MatLike] = det.draw_regions(regions)
        ff.save_images(img_draws,cv2Const=cv2.COLOR_BGR2RGB,path=IMAGES_PATH+"c_regioned/")
        
        #Agrupar las regiones:
        regions = det.groupped_images_regions
        img_draws: List[MatLike] = det.draw_regions(regions)
        ff.save_images(img_draws,cv2Const=cv2.COLOR_BGR2RGB,path=IMAGES_PATH+"d_groupped_regioned/")
                
        #Obtener, filtrar y dibujar las regiones en una copia:
        filter_regions: List[tuple[List[Rect],int]] = det.filter_images_regions
        img_draws = det.draw_regions(filter_regions)
        ff.save_images(img_draws,cv2Const=cv2.COLOR_BGR2RGB,path=IMAGES_PATH+"e_filter_regioned/")
        
        
        #Recortar las regiones y guardarlas:
        cropped_images: List[tuple[List[tuple[MatLike,Rect]],int]] = det.crop_regions(filter_regions)
        for crops,idx in cropped_images:
            imgs: List[MatLike] = [img for img,_ in crops]
            ff.save_images(imgs,cv2Const=cv2.COLOR_BGR2RGB,extra=f"{idx}-",path=IMAGES_PATH+"f_cropped/")
            
        #Aplicamos mascara:
        reg_subpanels: List[tuple[List[Rect],int]] = []
        cropped_mask_images: List[tuple[List[tuple[MatLike,Rect]],int]] = det.apply_filter_cropped(cropped_images)
        for crops_mask,idx in cropped_mask_images:
            imgs = [img for img,_ in crops_mask]
            reg_subpanels.append(([reg for _,reg in crops_mask],idx))
            ff.save_images(imgs,extra=f"{idx}-",path=IMAGES_PATH+"g_cropped_mask/")
        
        #Pintamos las regiones finales:
        text = det.draw_final_regions(images_final_regioned,cropped_mask_images,nameFiles)
        ff.save_images(images_final_regioned,cv2Const=cv2.COLOR_BGR2RGB,path=IMAGES_PATH+"h_final_regioned/")

        #Creamos el txt con las regiones finales listadas:
        ff.create_txt(FILES_PATH+"exit.txt",text)
        
        #filtrar regiones y obtener paneles:
        final_croppeds:List[tuple[List[MatLike],int]]=[]
        for crops,idx in det.crop_regions(reg_subpanels):
            final_croppeds.append(([img for img,_ in crops],idx))
            
        #Guardar los recortes finales:
        for imgs,idx in final_croppeds:
            ff.save_images(imgs,cv2Const=cv2.COLOR_BGR2RGB,extra=f"{idx}-",path=IMAGES_PATH+"i_final_cropped/")
            
        return final_croppeds
    
    def exec_normalizer_test(
        self,
        images: List[tuple[List[MatLike],int]]
    )->List[MatLike]:
        
        ff.remove_directory_content(IMAGES_PATH+"j_improve_images/")
        
        images_temp:List[MatLike]=[]
        for imgs,_ in images:
            for img in imgs:
                images_temp.append(img)
        images = images_temp
        
        nor = Normalizer(images)
        nor.clahe_apply()
        nor.corregir_perspectiva()
        
        ff.save_images(nor.images,IMAGES_PATH+"j_improve_images/",cv2Const=cv2.COLOR_BGR2RGB)