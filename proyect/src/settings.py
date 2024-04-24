import os

#---FileFuncs:
GLOBAL_PATH = os.path.abspath(__file__).replace("\\settings.py","")+"/../../"
IMAGES_PATH = GLOBAL_PATH + "proyect/images/"


#---Detector:

#MSER:
DELTA = 4
MIN_AREA = 1000
MAX_AREA = 80000
MAX_VARIATION = 0.9
MIN_DIVERSITY = 0.1

#Group:
GROUP_THRESHOLD = 1
EPS = 0.07

#Filter:
MIN_RATIO = 0.4
MAX_RATIO = 4
MAX_WIDTH = 500
MAX_HEIGHT = 500 
MIN_WIDTH = 50
MIN_HEIGHT = 50
ENLARGE_WIDTH = 5
ENLARGE_HEIGHT = 5

#Regions Drawer:
COLOR_BORDER = (0,255,0)
THICKNESS = 2

#Cropped:
CROPPED_TAM = (200,100)