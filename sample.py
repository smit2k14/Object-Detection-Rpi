# USAGE
# python real_time_object_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
from imutils.object_detection import non_max_suppression
import pytesseract
import numpy as np

import argparse
import imutils
import time
import cv2
from gtts import gTTS
from playsound import playsound
import os

from collections import defaultdict, deque
import text_rec_new as trn
import time

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

DIM = [[690,2310], [40,50], [3,6], [5*12,25*12], [8, 2.5], [150,100], [56.5, 70], [10, 15], [36.5, 21.57], [65,100], [45, 35], [25, 30], [81, 92], [23,38], [68, 18], [9,12], [43,50], [32, 91], [100,80], [18,32]]

avg_dim = defaultdict(list)
for i in range(len(CLASSES)):
	if i != 0:
		avg_dim[CLASSES[i]] = DIM[i-1]

FOCAL_LENGTH = 0.144

def distance_to_camera(knownDim, calculatedDim, focal_length = FOCAL_LENGTH):
    if(knownDim[0]> knownDim[1]):
        return ((knownDim[0] * focal_length)/calculatedDim[0])
    else:
        return ((knownDim[1] * focal_length)/calculatedDim[1])


COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start()

time.sleep(1.0)
fps = FPS().start()

# Making queues for calculating moving average of depth for each class
classes_queues = dict()
for i in range(len(CLASSES)):
	classes_queues[CLASSES[i]] = deque([0])

moving_avg_count = 5

# For gTTS
current_queue = deque()
classes_counters = defaultdict(int)

check_value = 10

# loop over the frames from the video stream
while True:

	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	results = trn.get_texts(frame)

	if len(results) != 0:
		for ((startX, startY, endX, endY), text) in results:
			print(text)
	# grab the frame dimensions and convert it to a blob
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
		0.007843, (300, 300), 127.5)

	# pass the blob through the network and obtain the detections and
	# predictions
	net.setInput(blob)
	detections = net.forward()

	# loop over the detections
	for i in np.arange(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the prediction
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the `confidence` is
		# greater than the minimum confidence
		if confidence > args["confidence"]:
			# extract the index of the class label from the
			# `detections`, then compute the (x, y)-coordinates of
			# the bounding box for the object
			idx = int(detections[0, 0, i, 1])
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			#print(f'startX-{startX}\nstartY-{startY}\nendX-{endX}\nendY-{endY}')
			height = (endY-startY)
			width = (endX-startX)
			#print(f'Height-{height}, Width-{width}')

			# draw the prediction on the frame
			label = "{}: {:.2f}%".format(CLASSES[idx],
				confidence * 100)

			print(CLASSES[idx])
			depth = distance_to_camera([height,width], avg_dim[CLASSES[idx]])

			# Calculating depth moving average

			if len(classes_queues[CLASSES[idx]]) < moving_avg_count:
				classes_queues[CLASSES[idx]].append(depth*96)
			else:
				classes_queues[CLASSES[idx]].popleft()
				classes_queues[CLASSES[idx]].append(depth*96)

			avg = 0
			if len(classes_queues[CLASSES[idx]]) == moving_avg_count:

				for k in range(moving_avg_count):
					avg += classes_queues[CLASSES[idx]][k]


			if len(current_queue) >= 1:
				current_queue.popleft()
				current_queue.append(CLASSES[idx])
			else:
				current_queue.append(CLASSES[idx])


			if classes_counters[CLASSES[idx]] == 0 or CLASSES[idx] not in current_queue:
				'''
				# Play detected object
				tts = gTTS(text=CLASSES[idx], lang='en')
				file = str("audio.mp3")
				tts.save(file)
				playsound(file)
				os.remove(file)
				'''
				print(f'Class-{CLASSES[idx]},Depth-{avg/moving_avg_count}\n')

			classes_counters[CLASSES[idx]] += 1

			if classes_counters[CLASSES[idx]] == check_value:
				classes_counters[CLASSES[idx]] = 0


			cv2.rectangle(frame, (startX, startY), (endX, endY),
				COLORS[idx], 2)
			y = startY - 15 if startY - 15 > 15 else startY + 15
			cv2.putText(frame, label, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

	# update the FPS counter
	fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
