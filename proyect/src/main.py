from classes.Tester import Tester
import common.FileFuncs as ff
from settings import *

if __name__ == "__main__":
    
    images = ff.read_images()
    tester: Tester = Tester()
    tester.exec_general_test(images)