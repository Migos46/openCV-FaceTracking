import numpy as np
import cv2

path = "/Users/mike/Documents/PythonProjects/Exercise Files/Ch04/04_05 Face Detection/haarcascade_frontalface_default.xml"
cap = cv2.VideoCapture(0)

while(True):
	ret, frame = cap.read()
	#Frame color to gray
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	
	face_cascade = cv2.CascadeClassifier(path)

	faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05,minNeighbors=20,minSize=(40,40))
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