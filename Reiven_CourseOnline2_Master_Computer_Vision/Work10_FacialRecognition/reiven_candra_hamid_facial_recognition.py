import cv2 as cv
import numpy as np
from os import listdir
from os.path import isfile, join

##################### CREATE TRAINING DATA #####################

face_classifier = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

# function for detecting faces and returns the cropped one
# if no faces are detected, returns the input image
def face_extractor(image):
	gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
	faces = face_classifier.detectMultiScale(gray, 1.3, 5)
	
	if faces is ():
		return None
		
	# crop the faces
	for x, y, w, h in faces:
		cropped_face = image[y:y+h, x:x+w]
		
	return cropped_face
	
cam = cv.VideoCapture(0)	
count = 0	
	
# collect 100 self face sample from webcam
while True:
	ret, frame = cam.read()
	if face_extractor(frame) is not None: # consider there is a face detected
		count += 1
		face = cv.resize(face_extractor(frame), (200, 200))
		face = cv.cvtColor(face, cv.COLOR_BGR2GRAY)
	
		# save file
		file_path = './self_face/' + str(count) + '.jpg'
		cv.imwrite(file_path,face)
	
		# display current count
		cv.putText(face, str(count), (50, 50), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
		cv.imshow('Extracting Image', face)
	
	else:
		print('Face not found')
		pass
		
	if cv.waitKey(1) == 13 or count == 100:
		break

cam.release()
cv.destroyAllWindows()
print('Collecting Samples Complete')
	
##################### END OF FIRST STEP #####################	

##################### TRAIN MODEL #####################
	
data_path = './self_face/'	
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]	
	
# create array for training data and labels
train_data, label = [], []

# open training images and create numpy array for training data
for i, files in enumerate(onlyfiles):
	images_path = data_path + onlyfiles[i]
	images = cv.imread(images_path,cv.IMREAD_GRAYSCALE)
	train_data.append(np.asarray(images, dtype = np.uint8))
	label.append(i)
	
# create numpy array for both training data and label
label = np.asarray(label, dtype = np.int32)	
	
# initialize facial recognizer
model = cv.face.LBPHFaceRecognizer_create()
	
# train the model
model.train(np.asarray(train_data),np.asarray(label))	
print('Training Model')
	
##################### END OF SECOND STEP #####################	
	
##################### TEST FACIAL RECOGNITION #####################	
	
def face_detector(image, size = 0.5):
	gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
	faces = face_classifier.detectMultiScale(gray, 1.3, 5)
	
	if faces is ():
		return image, []
	
	for x, y, w, h in faces:
		cv.rectangle(image, (x, y), (x+w,y+h), (0, 255, 255), 2)
		roi = image[y:y+h, x:x+w]
		roi = cv.resize(roi, (200, 200))
	
	return image,roi
	
cam = cv.VideoCapture(0)	
	
while True:
	ret,frame = cam.read()
	image, face = face_detector(frame)
	
	try:
		face = cv.cvtColor(face,cv.COLOR_BGR2GRAY)
		
		# pass face to prediction model
		results = model.predict(face)
		
		# results comprises of a tuple containing label and confidence value
		if results[1] < 500:
			confidence = int(100 * (1 - (results[1])/400))
			conf_percent = str(confidence) + ' % confidence it is the user'
		cv.putText(image, conf_percent, (100, 120), cv.FONT_HERSHEY_COMPLEX, 1, (255, 120, 150), 2)
		
		if confidence > 75:
			cv.putText(image, 'Unlocked', (250, 450), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
			cv.imshow('Reiven Face Recognition', image)
			
		else:
			cv.putText(image, 'Locked', (250, 450), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
			cv.imshow('Reiven Face Recognition', image)
	
	except:
		cv.putText(image, 'No Face Found', (220, 120), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)
		cv.putText(image, 'Locked', (250, 450), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
		cv.imshow('Reiven Face Recognition', image)
	
	if cv.waitKey(1) == 13:
		break
##################### END OF THIRD STEP #####################

cam.release()
cv.destroyAllWindows()	