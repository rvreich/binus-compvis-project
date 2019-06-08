import cv2 as cv
import numpy as np

# import haarcascade classifier for full body detection
body_classifier = cv.CascadeClassifier('haarcascade_fullbody.xml')
car_classifier = cv.CascadeClassifier('haarcascade_car.xml')

cam = cv.VideoCapture('walking.avi')

while cam.isOpened():
	ret, frame = cam.read()
	gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
	
	bodies = body_classifier.detectMultiScale(gray, 1.2, 3)
	
	for x,y,w,h in bodies:
		cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
		cv.imshow('Pedestrians', frame)
		
	if cv.waitKey(1) == 13:
		break

cam.release()		
		
cam = cv.VideoCapture('cars.avi')

while cam.isOpened():
	ret, frame = cam.read()
	gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
	
	car = car_classifier.detectMultiScale(gray, 1.3, 3)
	
	for x,y,w,h in car:
		cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
		cv.imshow('Cars', frame)
		
	if cv.waitKey(1) == 13:
		break
		
cam.release()
cv.destroyAllWindows();