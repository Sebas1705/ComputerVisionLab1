import os
from typing import List
import cv2
from cv2.typing import MatLike
import shutil
from settings import IMAGES_PATH


def save_images(
    images_to_save: List[MatLike],
    path:str,
    extra:str="",
    cv2Const:int=None
) -> None:
    """
    Saves a list of images to the specified path.

    Parameters:
    -----------
    images_to_save : List[MatLike] 
        A list of images to be saved.
    path : str
        The path to which the images are to be saved.
    extra : str, optional 
        An extra string to be appended to the file name. Defaults to "".
    cv2Const : int, optional
        An constant that transform an image into a colar format, Defaulst to None
    """
    for i in range(len(images_to_save)):
        img: MatLike = cv2.cvtColor(images_to_save[i],cv2Const) if cv2Const!=None else images_to_save[i]
        cv2.imwrite(f"{path}/{extra}{i:0>5}.png",img)
        
def read_images(
    path:str=IMAGES_PATH+"test_selected/",
    start:int=0,
    end:int=20,
    cv2Const:int=cv2.COLOR_BGR2RGB
) -> tuple[list[MatLike],list[str]]:
    """
    Reads a list of images from the specified path.

    Parameters:
    -----------
    path : str, optional
        The path from which the images are to be read.
        Defaults to "../../images/test".
    start : int, optional
        The index of the first image to be read.
        Defaults to 0.
    end : int, optional
        The index of the last image to be read.
        Defaults to 20.
    cv2Const : int, optional
        The constant used for converting the images.
        Defaults to cv2.COLOR_BGR2RGB.

    Returns:
    --------
    list[MatLike]
        A list of images read from the specified path.
    """
    return ([
        cv2.cvtColor(cv2.imread(os.path.join(path,file)),cv2Const) 
        if cv2Const is not None else cv2.imread(os.path.join(path,file))
        for file in os.listdir(path)
    ][start:end],
    [file for file in os.listdir(path)])

def remove_directory_content(
    path:str
) -> None:
    """
    Removes the content of a directory.

    Parameters:
    -----------
    path : str
        The path of the directory to be emptied.
    """
    for file in os.listdir(path):
        path_complete = os.path.join(path,file)
        if os.path.isfile(path_complete):
            os.remove(path_complete)
        elif os.path.isdir(path_complete):
            shutil.rmtree(path_complete)
            
def remove_images_dests() -> None:
    """
    Removes all the images from the destinations folders.
    """
    remove_directory_content(path=IMAGES_PATH+"a_gray_before/")
    remove_directory_content(path=IMAGES_PATH+"b_gray_after/")
    remove_directory_content(path=IMAGES_PATH+"c_regioned/")
    remove_directory_content(path=IMAGES_PATH+"d_groupped_regioned/")
    remove_directory_content(path=IMAGES_PATH+"e_filter_regioned/")
    remove_directory_content(path=IMAGES_PATH+"f_cropped")
    remove_directory_content(path=IMAGES_PATH+"g_cropped_mask/")
    remove_directory_content(path=IMAGES_PATH+"h_final_regioned/")
    remove_directory_content(path=IMAGES_PATH+"i_final_cropped/")

def create_txt(path:str,text:str):    
    """
    Creates a text file with the given text.

    Parameters:
    -----------
    path : str
        The path of the file to be created.
    text : str
        The text to be written in the file.
    """
    if os.path.exists(path):
        os.remove(path)    
    with open(path, 'w') as file:
        file.write(text)