import cv2
import imutils
import numpy as np

print("[INFO] loading model...")
prototxt = 'deploy.prototxt'
model = 'res10_300x300_ssd_iter_140000.caffemodel'
net = cv2.dnn.readNetFromCaffe(prototxt, model)

def extract(image):
	# resize it to have a maximum width of 400 pixels
	(h, w) = image.shape[:2]
	# image = imutils.resize(image, width=400)
	# resize it to have a maximum width of 400 pixels
	# image = imutils.resize(image, width=400)
	blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
	print("[INFO] computing object detections...")
	net.setInput(blob)
	detections = net.forward()
	images = []
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with the prediction
		confidence = detections[0, 0, i, 2]
		
		# filter out weak detections by ensuring the `confidence` is
		# greater than the minimum confidence threshold
		if confidence > 0.7:
			# compute the (x, y)-coordinates of the bounding box for the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			# draw the bounding box of the face along with the associated probability
			text = "{:.2f}%".format(confidence * 100)
			y = startY - 10 if startY - 10 > 10 else startY + 10
			images.append(image[startY:endY,startX:endX])
	print("len faces :",len(images))
	# print(images)
	return images,image



import sys
import os

if len(sys.argv)==2 and sys.argv[1] == "run":
	data = os.listdir('dataset_faces')
	for i in data:
		image_file = 'dataset_faces/'+i
		im = cv2.imread(image_file)
		tmp = extract(im)[0][0]
		cv2.imwrite('cropped_dataset_faces/'+i,)