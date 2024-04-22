import cv2
from matplotlib import pyplot as plt 
import os
import numpy as np

def save_images(images_to_save):
    for i in range(len(images_to_save)):
        cv2.imwrite(f"./imagenesResultado/{i:0>5}.png",cv2.cvtColor(images_to_save[i],cv2.COLOR_BGR2RGB))

def save_images2(images_to_save):
    for i in range(len(images_to_save)):
        cv2.imwrite(f"./imagenesfinales/{i:0>5}.png",cv2.cvtColor(images_to_save[i],cv2.COLOR_BGR2RGB))
        
directory = "./imagenesTest/"
rgb_images = [cv2.cvtColor(cv2.imread(os.path.join(directory, file)),cv2.COLOR_BGR2RGB) for file in os.listdir(directory)][0:10]
imgs = [cv2.cvtColor(cv2.imread(os.path.join(directory, file)),cv2.COLOR_BGR2RGB) for file in os.listdir(directory)][0:10]


plt.subplot(1,2,1)
plt.imshow(rgb_images[0])
plt.title("Primera")

plt.subplot(1,2,2)
plt.imshow(rgb_images[len(rgb_images)-1])
plt.title("Ulitma")

gray_images = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in rgb_images]

plt.subplot(2,2,1)
plt.imshow(gray_images[0], cmap="gray")
plt.title("Primera Antes")

plt.subplot(2,2,2)
plt.imshow(gray_images[len(rgb_images)-1], cmap="gray")
plt.title("Ulitma Antes")

for i in range(len(gray_images)):
    gray_images[i] = cv2.equalizeHist(gray_images[i])

plt.subplot(2,2,3)
plt.imshow(gray_images[0], cmap="gray")
plt.title("Primera Despues")

plt.subplot(2,2,4)
plt.imshow(gray_images[len(rgb_images)-1], cmap="gray")
plt.title("Ulitma Despues")

mser = cv2.MSER_create(delta=4, min_area=1000, max_area=80000, max_variation=0.9, min_diversity=0.1)

def getRegions(grays):
    allRegions = []
    for gray in grays:
        _, boundingBoxes = mser.detectRegions(gray)
        allRegions.append(boundingBoxes)
    return allRegions

def groupRegions(list_boxes):
    groupped_regions = []
    for boxes in list_boxes:
        groupped_rects, _ = cv2.groupRectangles(boxes, 1, eps=0.07)
        groupped_regions.append(groupped_rects)        
    return groupped_regions

def drawRegions(images_list, all_regions):   
    groupped_regions = []
    i = 0
    for img, regs in zip(images_list, all_regions):   
        for region in regs:            
            x, y, w, h = region
            aspect_ratio = float(w) / h
            if 0.5 < aspect_ratio < 4:
                if w < 500:
                    if h < 500:
                        area = h * w
                        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
                        groupped_regions.append((region,i)) 
        i = i +1  
    return groupped_regions  


dectected_regions = drawRegions(rgb_images, groupRegions(getRegions(gray_images)))

plt.subplot(1,2,1)
plt.imshow(rgb_images[0])
plt.title("Primera")

plt.subplot(1,2,2)
plt.imshow(rgb_images[len(rgb_images)-1])
plt.title("Ulitma")

# save_images(rgb_images)

recortes = []
for region in dectected_regions:    
    x, y, w, h = region[0]    
    cropped_image = rgb_images[region[1]][y:y+h, x:x+w]
    recortes.append((cropped_image,region[1],region[0]))

plt.subplot(1,2,1)
plt.imshow(recortes[len(recortes)-1][0])
plt.title("Primera")

plt.subplot(1,2,2)
plt.imshow(rgb_images[recortes[len(recortes)-1][1]])
plt.title("Imagen")

hsv_images = [cv2.cvtColor(im, cv2.COLOR_BGR2HSV) for im in rgb_images]

rgb_recortes = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img,i,r in recortes]
hsv_recortes = [cv2.cvtColor(im, cv2.COLOR_BGR2HSV) for im in rgb_recortes]

lower_blue = np.array([100,50,60])
upper_blue = np.array([130,255,255])

def apply_mask(img, lower_blue, upper_blue):
    mask_aux  = cv2.inRange(img, lower_blue, upper_blue)
    return cv2.bitwise_and(img, img, mask=mask_aux)

def save_recortes(images_with_index,mask: bool):
    for i in range(len(images_with_index)):
        cv2.imwrite(f"./recortes/{i:0>5}-{images_with_index[i][1]}{"" if mask else "-noMask"}.png",cv2.cvtColor(images_with_index[i][0],cv2.COLOR_BGR2RGB))   


plt.subplot(2,2,1)
plt.imshow(recortes[46][0])
plt.subplot(2,2,2)
plt.imshow(rgb_images[3])
plt.subplot(2,2,3)
plt.imshow(cv2.cvtColor(cv2.cvtColor(apply_mask(hsv_recortes[46], lower_blue, upper_blue), cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2RGB))
plt.subplot(2,2,4)
plt.imshow(cv2.cvtColor(cv2.cvtColor(apply_mask(hsv_images[3], lower_blue, upper_blue), cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2RGB))            


recortes_mascara = []
for rec in hsv_recortes:
    recortes_mascara.append(cv2.cvtColor(cv2.cvtColor(apply_mask(rec, lower_blue, upper_blue), cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2RGB))


imgs_noMask = []
imgs_mask = []
region_blue = []
for i in range(len(recortes_mascara)):
    gris = cv2.cvtColor(recortes_mascara[i], cv2.COLOR_BGR2GRAY)
    # Contar los píxeles negros (píxeles con valor 0)
    total_pixeles = gris.shape[0] * gris.shape[1]
    pxeles_negros = total_pixeles - cv2.countNonZero(gris)
    # Calcular el porcentaje de píxeles negros
    porcentaje = (pxeles_negros / total_pixeles) * 100
    if porcentaje < 14:
        imgs_noMask.append(recortes[i])
        imgs_mask.append((recortes_mascara[i],recortes[i][1]))
        region_blue.append((recortes[i][2],recortes[i][1]))

print(region_blue)
save_recortes(imgs_noMask, False)
save_recortes(imgs_mask, True)

def draw_final_region(region,img):
        for reg,i in region:
            x, y, w, h = reg
            cv2.rectangle(img[i], (x,y), (x+w, y+h), (0,255,0), 2)            


draw_final_region(region_blue,imgs)
save_images2(imgs)
