import numpy as np
import cv2

path = "/Users/mike/Documents/PythonProjects/Exercise Files/Ch04/04_05 Face Detection/haarcascade_frontalface_default.xml"
cap = cv2.VideoCapture(0)

while(True):
	ret, frame = cap.read()
	#Frame color to gray
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	
	face_cascade = cv2.CascadeClassifier(path)

	# Parameters detectMultiScale:	
	# 	cascade – Haar classifier cascade (OpenCV 1.x API only). It can be loaded from XML or YAML file using Load(). When the cascade is not needed anymore, release it using cvReleaseHaarClassifierCascade(&cascade).
	# 	image – Matrix of the type CV_8U containing an image where objects are detected.
	# 	objects – Vector of rectangles where each rectangle contains the detected object.
	# 	scaleFactor – Parameter specifying how much the image size is reduced at each image scale.
	# 	minNeighbors – Parameter specifying how many neighbors each candidate rectangle should have to retain it.
	# 	flags – Parameter with the same meaning for an old cascade as in the function cvHaarDetectObjects. It is not used for a new cascade.
	# 	minSize – Minimum possible object size. Objects smaller than that are ignored.
	# 	maxSize – Maximum possible object size. Objects larger than that are ignored.
	faces = face_cascade.detectMultiScale(gray, scaleFactor=1.15,minNeighbors=20,minSize=(40,40))
	#print(len(faces))

	for (x, y, w, h) in faces:
		cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
		cv2.putText(frame, "Face detected", (x, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 2,(0, 255,0), 10)

	cv2.imshow("Frame", frame)

	#Register a waitKey  with a value of 1,
	#this will indicate that it will run every 1 ms
	#Try different numbers and check what's happening
	ch = cv2.waitKey(1)	
	if ch & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()