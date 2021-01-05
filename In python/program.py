import numpy as np
import cv2
import math

def angle(s,e,f):
    v1 = [s[0]-f[0],s[1]-f[1]]
    v2 = [e[0]-f[0],e[1]-f[1]]
    ang1 = math.atan2(v1[1],v1[0])
    ang2 = math.atan2(v2[1],v2[0])
    ang = ang1 - ang2
    if (ang > np.pi):
        ang -= 2*np.pi
    if (ang < -np.pi):
        ang += 2*np.pi
    return ang*180/np.pi


cap = cv2.VideoCapture(0)
backSub = cv2.createBackgroundSubtractorMOG2(detectShadows = True)

if not cap.isOpened:
    print ("Unable to open file")
    exit(0)
pt1 = (400,100)
pt2 = (600,300)

l_rate = -1

while (True):
	ret,frame=cap.read()
	if not ret:
		exit(0)

	frame = cv2.flip(frame,1)

	roi = frame[pt1[1]:pt2[1],pt1[0]:pt2[0],:].copy() # Muestra todo lo que tenga el rectangulo que se encuentra en frame en ROI

	fgMask = backSub.apply(roi, learningRate = l_rate) # Aplicamos la mascara al fondo
	contours, hierarchy = cv2.findContours(fgMask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2:]
	
	if len(contours) != 0:
		larguestContour = max(contours, key=cv2.contourArea) # mayor contorno
		cv2.drawContours(roi, larguestContour, -1, (0,255,0),3)

		x,y,w,h = cv2.boundingRect(larguestContour)
		aspect_ratio = float(w)/h
		pt3 = (x,y)
		pt4 = (x+w,y+h)
		cv2.rectangle(roi,pt3,pt4,(0,0,255),1)

		hull = cv2.convexHull(larguestContour, returnPoints = False) #Malla convexa
		defects = cv2.convexityDefects(larguestContour,hull)

		if defects is not None:
			for i in range(len(defects)):
				s,e,f,d = defects[i,0]
				start = tuple(larguestContour[s][0])
				end = tuple(larguestContour[e][0])
				far = tuple(larguestContour[f][0])
				depth = d/256.0
				print(depth)
				ang = angle(start,end,far)
				if 0.3*h < depth and ang < 90:
					cv2.line(roi,start,end,[255,0,0],2)
					cv2.circle(roi,far,5,[0,0,255],-1)

	cv2.rectangle(frame,pt1,pt2,(255,0,0))
	cv2.imshow('frame',frame)
	cv2.imshow('ROI',roi)
	cv2.imshow('Foreground Mask',fgMask)

	keyboard = cv2.waitKey(40)
	if keyboard & 0xFF == ord('q'):
		break
	if keyboard & 0xFF == ord('v'):
		l_rate = 0

cap.release()
cv2.destroyAllWindows()


