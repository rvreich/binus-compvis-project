import cv2 as cv
import numpy as np

cam = cv.VideoCapture(0)

# define range of purple color in HSV
lower_purple = np.array([130, 50, 90])
upper_purple = np.array([170, 255, 255])

# create empty points array
points = []

# get default camera window size
ret, frame = cam.read()
height, width = frame.shape[:2]
frame_count = 0

while True:

    ret, frame = cam.read()
    hsv_img = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # Threshold the HSV image to get only blue colors
    mask = cv.inRange(hsv_img, lower_purple, upper_purple)
    
    contours, _ = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    # Create empty centre array to store centroid center of mass
    center =   int(height/2), int(width/2)

    if len(contours) > 0:
        
        # Get the largest contour and its center 
        c = max(contours, key = cv.contourArea)
        (x, y), radius = cv.minEnclosingCircle(c)
        M = cv.moments(c)
        try:
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        except:
            center =   int(height/2), int(width/2)

        if radius > 25:
            
            # Draw cirlce and leave the last center creating a trail
            cv.circle(frame, (int(x), int(y)), int(radius), (0, 0, 255), 2)
            cv.circle(frame, center, 5, (0, 255, 0), -1)
            
    # Log center points 
    points.append(center)
    
    # loop over the set of tracked points
    if radius > 25:
        for i in range(1, len(points)):
            try:
                cv.line(frame, points[i - 1], points[i], (0, 255, 0), 2)
            except:
                pass
        frame_count = 0
    else:
        # Count frames 
        frame_count += 1
        
        if frame_count == 10:
            points = []
            frame_count = 0
            
    # Display our object tracker
    frame = cv.flip(frame, 1)
    cv.imshow("Reiven Object Tracker", frame)

    if cv.waitKey(1) == 13:
        break

cam.release()
cv.destroyAllWindows()