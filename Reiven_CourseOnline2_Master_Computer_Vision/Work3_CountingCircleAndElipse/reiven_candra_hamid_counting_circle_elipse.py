import cv2 as cv
import numpy as np

img_1 = cv.imread('circle_elipse_1_11_17.png', 0)
cv.imshow('Original Image 1', img_1)
cv.waitKey(0)

img_2 = cv.imread('circle_elipse_2_5_1.png', 0)
cv.imshow('Original Image 2', img_2)
cv.waitKey(0)

# initialize a detector
detector = cv.SimpleBlobDetector_create()

# detect the blob
keypoints_1 = detector.detect(img_1)
keypoints_2 = detector.detect(img_2)

# draw blobs on image using red circle mark
blank = np.zeros((1, 1))
blobs_1 = cv.drawKeypoints(img_1, keypoints_1, blank, (0, 0, 255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
blobs_2 = cv.drawKeypoints(img_2, keypoints_2, blank, (0, 0, 255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

num_blobs = len(keypoints_1)
text_1 = "Total Blobs on Image 1: " + str(len(keypoints_1))
cv.putText(blobs_1, text_1, (20, 400), cv.FONT_HERSHEY_SIMPLEX, 1, (100, 0, 255), 2)
text_2 = "Total Blobs on Image 2: " + str(len(keypoints_2))
cv.putText(blobs_2, text_2, (20, 400), cv.FONT_HERSHEY_SIMPLEX, 1, (100, 0, 255), 2)

cv.imshow('Blobs using default params 1', blobs_1)
cv.imshow('Blobs using default params 2', blobs_2)
cv.waitKey(0)

# set filtering parameter
params = cv.SimpleBlobDetector_Params()

# set area filtering parameter
params.filterByArea = True
params.minArea = 100

# set circularity filtering parameter
params.filterByCircularity = True
params.minCircularity = 0.9

# set convexity filtering parameter
params.filterByConvexity = False
params.minConvexity = 0.2

# set inertia filtering parameter
params.filterByInertia = True
params.minInertiaRatio = 0.01

detector = cv.SimpleBlobDetector_create(params)

keypoints_1v2 = detector.detect(img_1)
keypoints_2v2 = detector.detect(img_2)

blank = np.zeros((1, 1))
blobs_1v2 = cv.drawKeypoints(img_1, keypoints_1v2, blank, (0, 0, 255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
blobs_2v2 = cv.drawKeypoints(img_2, keypoints_2v2, blank, (0, 0, 255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

num_blobs = len(keypoints_1)
text_1v2 = "Optimized Blobs Img 1: " + str(len(keypoints_1))
cv.putText(blobs_1v2, text_1v2, (20, 400), cv.FONT_HERSHEY_SIMPLEX, 1, (100, 0, 255), 1)
text_2v2 = "Optimized Blobs Img 2: " + str(len(keypoints_2))
cv.putText(blobs_2v2, text_2v2, (20, 400), cv.FONT_HERSHEY_SIMPLEX, 1, (100, 0, 255), 1)

cv.imshow('Blobs using optimized params 1', blobs_1v2)
cv.imshow('Blobs using optimized params 2', blobs_2v2)
cv.waitKey(0)

cv.destroyAllWindows()