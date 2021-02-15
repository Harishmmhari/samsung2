import cv2
import numpy as np

#Right now I'm just testing out the ORB detection algorithm because I head that it is marginally faster than SIFT or SURF


cap = cv2.VideoCapture(0)
ret, frame = cap.read()
ret, last = cap.read()

frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
last = cv2.cvtColor(last, cv2.COLOR_BGR2GRAY) 

orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

kp, des = orb.detectAndCompute(frame, None)
kp_last = kp
des_last = des


while(True):
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    #Do the work here!
    
    #Lets see how long the orb algorithm actually takes to calculate
    #as long as we store the previous keypoints found we actually only ever need to run the detection once a frame
    
    kp, des = orb.detectAndCompute(frame, None)
    #print(des[0])
    img2 = cv2.drawKeypoints(frame,kp, outImage = None, color=(0,255,0), flags=0)
    
    #Another importatant thing to check is how long the matching takes
    
    matches = bf.match(des, des_last)
    matches = sorted(matches, key = lambda x:x.distance)
    #print(matches)
    matches.reverse()
    
    #last we need to get the matched keypoints and find the average distance moved on them, however we do not know the distance of each point from the camera.
    #solvePNP can actually be a bit slow so lets try and see if we can take advantage of knowing the focal length
    
    #If we know the focal length we could potentially use the difference in lateral movement of the keypoints to estimate distance from the camera with a sufficient amount of points

    img3 = cv2.drawMatches(frame,kp, last,kp_last, matches[:10], None,flags=2)
    # Display the resulting frame
    
    cv2.imshow('frame', frame)
    cv2.imshow('kp',img2)

    cv2.imshow('matches',img3)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    last = frame.copy()
    kp_last = kp
    des_last = des
    
cap.release()
cv2.destroyAllWindows()
