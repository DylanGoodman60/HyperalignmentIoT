import cv2
import hypertools as hyp
import numpy as np

cap = cv2.VideoCapture(0)


# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    
    frame = cv2.cvtColor(cv2.resize(frame, (0, 0), None, .5, .5), cv2.COLOR_BGR2GRAY)
    res = hyp.align([np.array(frame), np.array(frame)])
    cv2.imshow('built-in', res[1]/255)

    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()
