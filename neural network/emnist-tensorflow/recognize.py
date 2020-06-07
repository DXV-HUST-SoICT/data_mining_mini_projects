from __future__ import absolute_import

from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json
from matplotlib import pyplot as plt

import cv2
import numpy as np
import sys, os

WORKING_DIR = './'
characters = ['0','1','2','3','4','5','6','7','8','9',
		  'A','B','C','D','E','F','G','H','I','J',
		  'K','L','M','N','O','P','Q','R','S','T',
		  'U','V','W','X','Y','Z']

def load_model():
	json_file = open(os.path.join(WORKING_DIR, 'results/model_0.0.1.json'), 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	loaded_model.load_weights(os.path.join(WORKING_DIR, 'results/model_0.0.1.h5'))
	print('Model successfully loaded')
	return loaded_model

def recognize(filepath):
	model = load_model()
	image = cv2.imread(filepath)

	height, width, depth = image.shape
	#resizing the image to find spaces better
	image = cv2.resize(image, dsize=(width*5,height*4), interpolation=cv2.INTER_CUBIC)
	#grayscale
	gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

	#binary
	ret, thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)

	#dilation
	kernel = np.ones((5,5), np.uint8)
	img_dilation = cv2.dilate(thresh, kernel, iterations=1)

	#adding GaussianBlur
	gsblur=cv2.GaussianBlur(img_dilation,(5,5),0)

	cv2.imwrite('1_preprocess.png', gsblur)

	#find contours
	ctrs, hier = cv2.findContours(gsblur.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	m = list()
	#sort contours
	sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
	dp = image.copy()
	for i, ctr in enumerate(sorted_ctrs):
		# Get bounding box
		x, y, w, h = cv2.boundingRect(ctr)
		height_pad = int(0.2*h)
		weight_pad = int(0.2*w)
		cv2.rectangle(dp,(x-weight_pad,y-height_pad),( x + w + weight_pad, y + h + height_pad ),(36,255,12), 9)
		
	cv2.imwrite('2_contours.png', dp)

	for i, ctr in enumerate(sorted_ctrs):
		# Get bounding box
		x, y, w, h = cv2.boundingRect(ctr)
		# Getting ROI
		height_pad = int(0.2 * h)
		weight_pad = int(0.2 * w)
		roi = image[y-height_pad:y+h+height_pad, x-weight_pad:x+w+weight_pad]
		try:
			roi = cv2.resize(roi, dsize=(64,64), interpolation=cv2.INTER_CUBIC)
		except:
			continue
		roi = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
		
		roi = np.array(roi)
		t = np.copy(roi)
		t = t / 255.0
		t = 1-t
		t = t.reshape(1,64,64,1)
		m.append(roi)
		pred = model.predict_classes(t)

		# cv2.rectangle(dp,(x-weight_pad,y-height_pad),( x + w + weight_pad, y + h + height_pad ),(36,255,12), 9)
		cv2.putText(dp, characters[pred[0]] , (x, y-height_pad) , cv2.FONT_HERSHEY_SIMPLEX, 3, (90,0,255),9)
		print(characters[pred[0]])
		
	cv2.imwrite('3_recognized.png', dp)
	print("recognized")

	plt.imshow(dp)
	plt.show()


if __name__ == '__main__':
	if len(sys.argv) > 1:
		file_location = sys.argv[1].strip()
		recognize(file_location)
	else:
		print('you have to pass a path of image as a argument')
