from imutils.video import FPS
import numpy as np
import argparse
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="",
	help="path to (optional) input video file")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applyong non-maxima suppression")
ap.add_argument("-o", "--output", type=str, default="",
	help="path to (optional) output video file")

args = vars(ap.parse_args())


#settigs for object detection
base_path="yolo-coco"

labelsPath = os.path.sep.join([base_path,"coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")
print(LABELS)
print(len(LABELS))
# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([base_path, "yolov3.weights"])
configPath = os.path.sep.join([base_path, "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# check if we are going to use GPU
'''if (0):
	# set CUDA as the preferable backend and target
	print("[INFO] setting preferable backend and target to CUDA...")
	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
	net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)'''

# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the width and height of the frames in the video file
W = None
H = None

print("[INFO] accessing video stream...")
vs = cv2.VideoCapture(args["input"] if args["input"] else 0)
writer = None
fps = FPS().start()
display=input("display")
# loop over frames from the video file stream

#settings for orb

cap = vs
ret, frame = cap.read()
ret, last = cap.read()

frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
last = cv2.cvtColor(last, cv2.COLOR_BGR2GRAY)

orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

kp, des = orb.detectAndCompute(frame, None)
kp_last = kp
des_last = des


while True:
	# read the next frame from the file
	(grabbed, frame) = vs.read()

	# if the frame was not grabbed, then we have reached the end
	# of the stream
	if not grabbed:
		break

	# if the frame dimensions are empty, grab them
	if W is None or H is None:
		(H, W) = frame.shape[:2]

	# construct a blob from the input frame and then perform a forward
	# pass of the YOLO object detector, giving us our bounding boxes
	# and associated probabilities
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
	net.setInput(blob)
	layerOutputs = net.forward(ln)


	#####
	framecr=frame
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# Do the work here!
	ret=grabbed
	# Lets see how long the orb algorithm actually takes to calculate
	# as long as we store the previous keypoints found we actually only ever need to run the detection once a frame

	kp, des = orb.detectAndCompute(frame, None)
	# print(des[0])
	img2 = cv2.drawKeypoints(frame, kp, outImage=None, color=(0, 255, 0), flags=0)

	# Another importatant thing to check is how long the matching takes

	matches = bf.match(des, des_last)
	matches = sorted(matches, key=lambda x: x.distance)
	# print(matches)
	matches.reverse()

	# last we need to get the matched keypoints and find the average distance moved on them, however we do not know the distance of each point from the camera.
	# solvePNP can actually be a bit slow so lets try and see if we can take advantage of knowing the focal length

	# If we know the focal length we could potentially use the difference in lateral movement of the keypoints to estimate distance from the camera with a sufficient amount of points

	img3 = cv2.drawMatches(frame, kp, last, kp_last, matches[:10], None, flags=2)
	# Display the resulting frame





	last = frame.copy()
	kp_last = kp
	des_last = des






	#####
	# initialize our lists of detected bounding boxes, confidences,
	# and class IDs, respectively
	boxes = []
	confidences = []
	classIDs = []

	# loop over each of the layer outputs
	for output in layerOutputs:
		# loop over each of the detections
		for detection in output:
			# extract the class ID and confidence (i.e., probability)
			# of the current object detection
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			# filter out weak predictions by ensuring the detected
			# probability is greater than the minimum probability
			if confidence > args["confidence"]:
				# scale the bounding box coordinates back relative to
				# the size of the image, keeping in mind that YOLO
				# actually returns the center (x, y)-coordinates of
				# the bounding box followed by the boxes' width and
				# height
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")

				# use the center (x, y)-coordinates to derive the top
				# and and left corner of the bounding box
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))

				# update our list of bounding box coordinates,
				# confidences, and class IDs
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)

	# apply non-maxima suppression to suppress weak, overlapping
	# bounding boxes
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
		args["threshold"])

	# ensure at least one detection exists
	if len(idxs) > 0:
		# loop over the indexes we are keeping
		for i in idxs.flatten():
			# extract the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])

			# draw a bounding box rectangle and label on the frame
			color = [int(c) for c in COLORS[classIDs[i]]]
			cv2.rectangle(framecr, (x, y), (x + w, y + h), color,1)
			text = "{}: {:.4f}".format(LABELS[classIDs[i]],
				confidences[i])
			cv2.putText(framecr, text, (x, y - 5),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

	# check to see if the output frame should be displayed to our
	# screen
	if int(display)> 0:
		# show the output frame
		'''cv2.imshow("Frame", framecr)
		print("frame",np.shape(framecr))

		#cv2.imshow('frame', frame)
		cv2.imshow('kp', img2)
		print("kp", np.shape(img2))

		cv2.imshow('matches', img3)
		print("matches", np.shape(img3))'''
		hor=np.concatenate((framecr, img2), axis=1)
		verti=np.concatenate((hor,img3), axis=0)

		cv2.imshow("all",verti)


		key = cv2.waitKey(1) & 0xFF

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break

	# if an output video file path has been supplied and the video
	# writer has not been initialized, do so now
	if args["output"] != "" and writer is None:
		# initialize our video writer
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 30,
			(frame.shape[1], frame.shape[0]), True)

	# if the video writer is not None, write the frame to the output
	# video file
	if writer is not None:
		writer.write(frame)

	# update the FPS counter
	fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

cap.release()
cv2.destroyAllWindows()