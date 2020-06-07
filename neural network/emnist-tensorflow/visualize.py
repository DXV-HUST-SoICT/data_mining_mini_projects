from __future__ import absolute_import

from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json
from matplotlib import pyplot as plt
from tensorflow.image import resize as tf_img_resize

import cv2
import numpy as np
import sys, os
from matplotlib import pyplot as plt
from mnist import MNIST
import json

WORKING_DIR = './'
characters = ['0','1','2','3','4','5','6','7','8','9',
	'A','B','C','D','E','F','G','H','I','J',
	'K','L','M','N','O','P','Q','R','S','T',
	'U','V','W','X','Y','Z']

mndata = MNIST('data')

def load_model():
	json_file = open(os.path.join(WORKING_DIR, 'results/model_0.0.1.json'), 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	loaded_model.load_weights(os.path.join(WORKING_DIR, 'results/model_0.0.1.h5'))
	print('Model loaded')
	return loaded_model

def load_test_data():
	test_images_full, test_labels_full = mndata.load(
		'data/emnist/emnist-byclass-test-images-idx3-ubyte',
		'data/emnist/emnist-byclass-test-labels-idx1-ubyte')
	X_test = np.array([e for i, e in enumerate(test_images_full[:10000]) if test_labels_full[i] < 36])
	y_test = np.array([e for e in test_labels_full[:10000] if e < 36])
	X_test = X_test.reshape(X_test.shape[0], 28, 28)
	for t in range(X_test.shape[0]):
		X_test[t] = np.transpose(X_test[t])
	X_test_resized = np.array([tf_img_resize(image.reshape(28, 28, 1), [64, 64]).numpy() for image in X_test])
	print('Data loaded')
	return X_test_resized, y_test

def predict(model, data):
	X = data[0]
	y_true = data[1]
	# print(X.shape)
	y_pred = model.predict_classes(X)
	# print(y_pred)
	# print(y_true[0:32])
	print('Predicted')
	return y_pred

def statistic(y_true, y_pred):
	num_tests = len(y_true)
	num_classes = len(characters)
	count = np.ndarray(shape=(num_classes, num_classes), dtype=int)
	for i in range(num_classes):
		for j in range(num_classes):
			count[i, j] = 0
	rate = np.ndarray(shape=(num_classes, num_classes), dtype=float)
	for i in range(num_tests):
		count[y_true[i], y_pred[i]] += 1
	result = dict()
	result['count'] = dict()
	result['rate'] = dict()
	for i in range(num_classes):
		for j in range(num_classes):
			rate[i, j] = float(count[i, j] / sum(count[i, :]))
			result['count'][characters[i] + '_' + characters[j]] = int(count[i, j])
			result['rate'][characters[i] + '_' + characters[j]] = rate[i, j]
	result_file = open('result.json', 'w')
	result_json = json.dumps(result, sort_keys=True, indent=4, separators=(',', ': '))
	result_file.write(result_json)
	result_file.close()

if __name__ == '__main__':
	model = load_model()
	X_test, y_test = load_test_data()
	y_pred = predict(model, data=(X_test, y_test))
	statistic(y_test, y_pred)