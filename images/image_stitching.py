#dependencies and required packages
from imutils import paths
import numpy as numpy
import argparse
import imutils
import cv2


#argument parser
ap = argparse.ArgumentParser() #Instantiate an ArgumentParser
ap.add_argument("-i", "--images", type=str, required=True, help="path to input directory of images to stitch")
ap.add_argument("-o", "--output", type=str, required=True, help="path to the output image")
args = vars(ap.parse_args())

print("Loading images... ")
inputPaths = sorted(list(paths.list_images(args["images"])))
inputImages = []

# loading each path given by the directory from user
# getting images from the list
for inputPath in inputPaths:
	image = cv2.imread(inputPath)
	inputImages.append(image)

print("Stitching Images... ")
stitcher = cv2.Stitcher_create()
(status, stitched) = stitcher.stitch(inputImages)
#catches the process' output and put them respectively to each variable

if status == 0:
	cv2.imwrite(args["output"], stitched)

	cv2.imshow("Stitched", stitched)
	cv2.waitKey(0)
else:
	print("[INFO] image stitching failed ({})".format(status))
#get paths 
#1 - detect Keypoints 

# extract local invariant descriptors using SURF

#Match descriptors between images

#use RANSAC Algorithm to estimate homography matrix using matched feature vectors

#applying a warping transformation