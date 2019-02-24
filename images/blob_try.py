# # Standard imports
# import cv2
# #	import tifffile as tiff
# import numpy as numpy;
 
# # Read image
# im = cv2.imread("1.tif", -1)

# retval, threshold = cv2.threshold(im, 100, 255, cv2.THRESH_BINARY)

# grayscaled = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

# retval2, threshold2, = cv2.threshold(grayscaled, 100, 255, cv2.THRESH_BINARY)
# gaus = cv2.adaptiveThreshold(grayscaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 155, 1)
# retval3, otsu = cv2.threshold(grayscaled, 125, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# #im2gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
# #ret, mask = cv2.threshold(im2gray, 75, 255, cv2.THRESH_BINARY_INV)

# cv2.imshow('gaus', gaus)
# cv2.imshow('t1', threshold)
# cv2.imshow('t2', threshold2)
# cv2.imshow('otsu', otsu)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
from imutils import paths
import numpy as numpy
import argparse
import imutils
import cv2
#import matplotlib.pyplot as plt

img = cv2.imread("/output/output.tiff", -1)
#grayscaled = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(img, (3, 3), 2)
h, w = img.shape[:2]

"""Morphological gradient"""

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
gradient = cv2.morphologyEx(blur, cv2.MORPH_GRADIENT, kernel)
grayscaled = cv2.cvtColor(gradient, cv2.COLOR_BGR2GRAY)

mask = numpy.zeros(gradient.shape[:2],numpy.uint8)

bgdModel = numpy.zeros((1,65),numpy.float64)
fgdModel = numpy.zeros((1,65),numpy.float64)

rect = (50,50,1024,1280)

cv2.grabCut(gradient,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
mask2 = numpy.where((mask==2)|(mask==0),0,1).astype('uint8')
gradient = img*mask2[:,:,numpy.newaxis]

# detector = cv2.SimpleBlobDetector()
# print(type(gradient))
 
# # Detect blobs.
# keypoints = detector.detect(gradient)
 
# # Draw detected blobs as red circles.
# # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
# im_with_keypoints = cv2.drawKeypoints(gradient, keypoints, numpy.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
 

cv2.imwrite("m_g_output.tiff", gradient);

# cv2.namedWindow('image',cv2.WINDOW_NORMAL)
# cv2.resizeWindow('image', 640,512)
# cv2.imshow('image', gradient)
# cv2.waitKey()