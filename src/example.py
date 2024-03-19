# Project: Panoramic View
# Author: Guillem Bibiloni Femenias

import cv2
import matplotlib.pyplot as plt
from panoramic import computePanoramicView


# Example


# Read images
# (INTRODUCE YOUR OWN IMAGE PATHS)
img1 = cv2.imread('imgs/f_parc_1.jpg')
img2 = cv2.imread('imgs/f_parc_2.jpg')
img3 = cv2.imread('imgs/f_parc_3.jpg')
img4 = cv2.imread('imgs/f_parc_4.jpg')
img5 = cv2.imread('imgs/f_parc_5.jpg')

# Change color space to RGB
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
img4 = cv2.cvtColor(img4, cv2.COLOR_BGR2RGB)
img5 = cv2.cvtColor(img5, cv2.COLOR_BGR2RGB)



# For n images
result = computePanoramicView([img1,img2,img3,img4,img5]) # Introduce has many images in the list as the scene has.


fig = plt.figure(figsize=(19.2, 10.8), tight_layout=True) # Plot at full screen (1920x1080)
plt.axis('off')
plt.imshow(result)
plt.show()

