import cv2 as cv
import numpy as np

# function that returns the number of SIFT matches between those params
def ORB_detector(image,template):
	img_1 = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
	img_2 = template
	
	orb = cv.ORB_create()
	
	# obtain keypoints and descriptors
	keypoints_1, descriptors_1 = orb.detectAndCompute(img_1, None)
	keypoints_2, descriptors_2 = orb.detectAndCompute(img_2, None)
	
	# create matcher
	bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck = True)
	
	matches = bf.match(descriptors_1, descriptors_2)
	
	# sort the matches based on distancee. Least distance is better
	matches = sorted(matches, key = lambda val: val.distance)
	
	return len(matches)

cam = cv.VideoCapture(0)		
	
target = cv.imread('candyjar.jpg',0)

while True:
	ret, frame = cam.read()
	height,width = frame.shape[:2]
	
	# define ROI (Region Of Interest) Box Dimensions
	top_left_x = int(width / 3)
	top_left_y = int((height / 2) + (height / 4))
	bot_right_x = int((width / 3) * 2)
	bot_right_y = int((height /2 ) - (height / 4))
	
	cv.rectangle(frame, (top_left_x, top_left_y), (bot_right_x, bot_right_y), 255, 3)
	
	# crop window of observation we had defined above
	cropped = frame[bot_right_y:top_left_y, top_left_x:bot_right_x]
	
	# flip frame orientation horizontally
	frame = cv.flip(frame, 1)
	
	matches = ORB_detector(cropped,target)
	
	cv.putText(frame, str(matches), (450, 450), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 1)
	
	# define threshold to indicate object detection
	threshold = 500
	
	if matches > threshold:
		cv.rectangle(frame, (top_left_x, top_left_y), (bot_right_x, bot_right_y), (0, 255, 0), 3)
		cv.putText(frame, 'Object Found', (50,50), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
		
	cv.imshow('Custom Object Detector', frame)
	if cv.waitKey(1) == 13:
		break
		
cam.release()
cv.destroyAllWindows()