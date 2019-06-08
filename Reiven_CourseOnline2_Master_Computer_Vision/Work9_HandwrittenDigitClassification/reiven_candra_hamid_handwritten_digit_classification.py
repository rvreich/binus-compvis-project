import cv2 as cv
import numpy as np

#################### DATA PREPARATION, TRAINING, EVALUATION ####################

image = cv.imread('digits.png')
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
small = cv.pyrDown(image)
cv.imshow('Digits Image', small)
cv.waitKey(0)
cv.destroyAllWindows()

# split image to 5000 cells, each 20x20 pix -> 4-dim array 50 x 100 x 20 x 20
cells = [np.hsplit(row, 100) for row in np.vsplit(gray, 50)]

x = np.array(cells)

# split the data to 2 part -> training set and test data set
train = x[:, :70].reshape(-1, 400).astype(np.float32) # size -> (3500, 400)
test = x[:, 70:100].reshape(-1, 400).astype(np.float32) # size -> (1500, 400)

# create label for train and test data
k = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
train_label = np.repeat(k, 350)[:, np.newaxis]
test_label = np.repeat(k, 150)[:, np.newaxis]

# initiate kNN, train data, then test it with test data for k = 3
knn = cv.ml.KNearest_create()
knn.train(train, cv.ml.ROW_SAMPLE, train_label)
ret, result, neighbors, distance = knn.findNearest(test, k = 3)

# check the accuracy of classification
matches = result == test_label
correct = np.count_nonzero(matches)
accuracy = correct * (100.0 / result.size)
print('Accuracy is : %.2f' %accuracy + '%')

#################### END OF FIRST STEP ####################

#################### DEFINING NECESSARY FUNCTIONS ####################

# function for finding the x centroid coordinates
def x_cord_contour(contour):
	if cv.contourArea(contour) > 10:
		M = cv.moments(contour)
		return (int(M['m10']/M['m00']))

# function for making dimensional square with necessary black pixel padding when needed
def makeSquare(not_square):
	BLACK = [0, 0, 0]
	img_dim = not_square.shape
	height = img_dim[0]
	width = img_dim[1]
	if height == width:
		square = not_square
		return square
	else:
		doublesize = cv.resize(not_square, (2 * width, 2 *height), interpolation = cv.INTER_CUBIC)
		height = height * 2
		width = width * 2
		if height > width:
			pad = int((height - width) / 2)
			doublesize_square = cv.copyMakeBorder(doublesize, 0, 0, pad, pad, cv.BORDER_CONSTANT, value = BLACK)
		else:
			pad = int((width - height) / 2)
			doublesize_square = cv.copyMakeBorder(doublesize, pad, pad, 0, 0, cv.BORDER_CONSTANT, value = BLACK)
	
	return doublesize_square

# function for resizing image to specific dimensions
def resize_to_pixel(dimensions, image):
	buffer_pix = 4
	dimensions = dimensions - buffer_pix
	squared = image
	r = float(dimensions) / squared.shape[1]
	dim = (dimensions, int(squared.shape[0] * r))
	resized = cv.resize(image, dim, interpolation = cv.INTER_AREA)
	img_dim2 = resized.shape
	height_r = img_dim2[0]
	width_r = img_dim2[1]
	BLACK = [0, 0, 0]
	if height_r > width_r:
		resized = cv.copyMakeBorder(resized, 0, 0, 0, 1, cv.BORDER_CONSTANT, value = BLACK)
	if height_r < width_r:
		resized = cv.copyMakeBorder(resized, 1, 0, 0, 0, cv.BORDER_CONSTANT, value = BLACK)
	p = 2
	ReSizedImg = cv.copyMakeBorder(resized, p, p, p, p, cv.BORDER_CONSTANT, value = BLACK)
	img_dim = ReSizedImg.shape
	height = img_dim[0]
	width = img_dim[1]
	return ReSizedImg
	
#################### END OF SECOND STEP ####################	
	
#################### LOAD IMAGE,PREPROCESSING, CLASSIFY DIGITS ####################	
	
target = cv.imread('numbers.jpg')	
tgray = cv.cvtColor(target,cv.COLOR_BGR2GRAY)	
	
cv.imshow('Target Image', target)
cv.imshow('Gray Target Image', tgray)
cv.waitKey(0)

# blur image, then use Canny to find edges
blurred = cv.GaussianBlur(tgray, (5, 5), 0)
cv.imshow('Blurred Target', blurred)
cv.waitKey(0)

edge = cv.Canny(blurred, 30, 150)
cv.imshow('Edged of Blurred Target', edge)
cv.waitKey(0)
	
# find contours	
contours,_ = cv.findContours(edge.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# sort contour using the x coordinates
filtered_contours = [c for c in contours if cv.contourArea(c) > 10]
contours = sorted(filtered_contours, key = x_cord_contour, reverse = False)

# create empty array for entire number storing
full_number = []

for c in contours:
	x, y, w, h = cv.boundingRect(c)
	
	if w >= 5 and h >= 25:
		roi = blurred[y:y+h, x:x+w]
		ret, roi = cv.threshold(roi, 127, 255, cv.THRESH_BINARY_INV)
		squared = makeSquare(roi)
		final = resize_to_pixel(20, squared)
		cv.imshow('Final Image', final)
		final_array = final.reshape((1, 400))
		final_array = final_array.astype(np.float32)
		ret, result, neighbors, dist = knn.findNearest(final_array, k = 1)
		number = str(int(float(result[0])))
		full_number.append(number)
		cv.rectangle(target, (x, y), (x+w, y+h), (0, 255, 255), 2)
		cv.putText(target, number, (x, y + 155), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)
		cv.imshow('Reiven Digit Detection',target)
		cv.waitKey(0)

#################### END OF THIRD STEP ####################

cv.destroyAllWindows()
print('The number is: ' + ''.join(full_number))