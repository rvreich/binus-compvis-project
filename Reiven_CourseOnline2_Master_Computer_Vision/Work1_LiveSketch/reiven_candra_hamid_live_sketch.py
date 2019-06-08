import cv2 as cv
import numpy as np

def sketch_image(frame):
	img_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
	img_gray_blur = cv.GaussianBlur(img_gray, (5,5), 0)
	canny_edges = cv.Canny(img_gray_blur, 10, 60)
	ret, mask = cv.threshold(canny_edges, 50, 255, cv.THRESH_BINARY_INV)
	return mask
	
capture = cv.VideoCapture(0)

while True:
	ret, frame = capture.read()
	cv.imshow('Reiven Live Sketch', sketch_image(frame))
	if cv.waitKey(1) == 13: #enter key is pressed
		break
		
capture.release()
cv.destroyAllWindows()