import cv2

#haar cascade classifier for frontal face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#haar cascade classifier for eye detection
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

def detect(gray,frame):
    faces = face_cascade.detectMultiScale(gray,1.3,5)
    for(x,y,w,h) in faces:
        #draw rectangle line
        #this here means draw rectangle in frame, from position x,y
        #to position (x+w),(y+h) with color Blue and line thickness of 2
        cv2.rectangle(frame, (x,y),(x+w,y+h),(255,0,0),2)
        #take all of the region of interest, and it to roi_gray and roi_color
        roi_gray = gray[y:y+h,x:x+w]
        roi_color = frame[y:y+h,x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray,1.1,20)
        for(ex,ey,ew,eh) in eyes:
			#draw rectangle line with the constriction of only inside the
			#region of interest inside the faces with color green and line
			#thickness of 2
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    return frame

#define video capture code
video_capture = cv2.VideoCapture(0)
while True:
    _,frame = video_capture.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    canvas = detect(gray,frame)
    cv2.imshow('Video',canvas)
	#if key 'q' was pressed, then break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
video_capture.release()
cv2.destroyAllWindows()