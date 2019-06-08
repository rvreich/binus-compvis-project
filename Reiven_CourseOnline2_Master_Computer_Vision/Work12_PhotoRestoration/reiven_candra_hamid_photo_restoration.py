import cv2 as cv
import numpy as np

image = cv.imread('abraham.jpg')
cv.imshow('Original Damaged Photo', image)
cv.waitKey(0)

marked_damages = cv.imread('mask.jpg', 0)
cv.imshow('Marked Damages', marked_damages)
cv.waitKey(0)

# make a mask out of our marked image by changing all colors 
# that are not white, to black
ret, thresh1 = cv.threshold(marked_damages, 254, 255, cv.THRESH_BINARY)
cv.imshow('Threshold Binary', thresh1)
cv.waitKey(0)

# dilate the marks since thresholding has narrowed it slightly
kernel = np.ones((7,7), np.uint8)
mask = cv.dilate(thresh1, kernel, iterations = 1)
cv.imshow('Dilated Mask', mask)
cv.imwrite("abraham_mask.png", mask)

cv.waitKey(0)
restored = cv.inpaint(image, mask, 3, cv.INPAINT_TELEA)

cv.imshow('Restored', restored)
cv.imwrite('Final_Photo.png', restored)
cv.waitKey(0)
cv.destroyAllWindows()
