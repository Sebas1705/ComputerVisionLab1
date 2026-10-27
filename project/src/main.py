from classes.Tester import Tester
import common.FileFuncs as ff
from typing import List
from cv2.typing import MatLike

if __name__ == "__main__":
    
    images,nameFiles = ff.read_images()
    tester: Tester = Tester()
    images: List[tuple[List[MatLike],int]] = tester.exec_general_test(images,nameFiles)
    
    tester.exec_normalizer_test(images)
    