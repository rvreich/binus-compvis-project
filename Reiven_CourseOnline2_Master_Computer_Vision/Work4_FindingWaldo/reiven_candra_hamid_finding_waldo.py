import cv2 as cv
import numpy as np

image = cv.imread('scene.jpg')
cv.imshow('Where is Waldo ?', image)
cv.waitKey(0)

gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

target = cv.imread('waldo.jpg', 0)

result = cv.matchTemplate(gray, target, cv.TM_CCOEFF)
min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)

# create a bounding box
top_left = max_loc
bot_right = (top_left[0] + 50, top_left[1] + 50)
cv.rectangle(image, top_left, bot_right, (0, 0, 255), 5)

cv.imshow('Where is Waldo?', image)
cv.waitKey(0)
cv.destroyAllWindows()