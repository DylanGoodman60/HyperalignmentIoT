# HyperalignmentIoT
Various scripts and diagrams used for "Hyper-alignment on SoM Edge Devices for Internet of Agriculture Things" academic paper. Work was done in a Summer 2022 USRA Co-op position

## Usage

Import library and open-cv, read images, align images, show result, save result

'''
import cv2
from hyp import *

img1 = cv2.imread('image1.jpg')
img2 = cv2.imread('image2.jpg')

aligned_result = align_RGB_images([img1.astype(float), img2.astype(float)])
show_image(aligned_result)
save_image("afilename.png", aligned_result)
'''
