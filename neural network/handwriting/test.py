import tensorflow as tf
AUTOTUNE = tf.data.experimental.AUTOTUNE

import IPython.display as display
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

# print(tf.__version__)

print('starting')
CLASS_NAME = np.array((os.listdir('./data/train')))
BATCH_SIZE = 32
IMG_HEIGHT = 64
IMG_WIDTH = 64

list_train = tf.data.Dataset.list_files('./data/train/*/*')
list_val = tf.data.Dataset.list_files('./data/val/*/*')
train_count = len(list(list_train))
val_count = len(list(list_val))

STEPS_PER_EPOCH = np.ceil(train_count/BATCH_SIZE)

def show_batch(image_batch, label_batch):
	plt.figure(figsize=(10, 10))
	for n in range(25):
		ax = plt.subplot(5, 5, n+1)
		plt.imshow(image_batch[n])
		plt.title(CLASS_NAME[label_batch[n]==1][0].title())
		plt.axis('off')
	plt.show()


def get_label(file_path):
	parts = tf.strings.split(file_path, os.path.sep)
	res = 0
	for idx in range(len(CLASS_NAME)):
		if CLASS_NAME[idx] == parts[-2]:
			res = idx
	# return CLASS_NAME.index(parts[-2] == CLASS_NAME)
	return res

def decode_img(img):
	img = tf.image.decode_jpeg(img, channels=3)
	img = tf.image.rgb_to_grayscale(img)
	img = tf.image.convert_image_dtype(img, tf.float32)
	return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT]) * (1. / 255)

def process_path(file_path):
	label = get_label(file_path)
	img = tf.io.read_file(file_path)
	img = decode_img(img)
	return img, label

print('reading data')
labeled_train_data = list_train.map(process_path, num_parallel_calls=AUTOTUNE)
labeled_val_data = list_val.map(process_path, num_parallel_calls=AUTOTUNE)

def prepare_for_training(ds, cache=True, shuffle_buffer_size=1000):
	if cache:
		if isinstance(cache, str):
			ds = ds.cache(cache)
		else:
			ds = ds.cache()

	ds = ds.shuffle(buffer_size=shuffle_buffer_size)
	# ds = ds.repeat()
	ds = ds.batch(BATCH_SIZE)
	ds = ds.prefetch(buffer_size=AUTOTUNE)

	return ds

print('preparing')
train_data = prepare_for_training(labeled_train_data)
val_data = prepare_for_training(labeled_val_data)

# image_batch, label_batch = next(iter(train_data))
# show_batch(image_batch.numpy(), label_batch.numpy())

# image_batch, label_batch = next(iter(val_data))
# show_batch(image_batch.numpy(), label_batch.numpy())

import time
default_timeit_steps = 1000

def timeit(ds, steps=default_timeit_steps):
	start = time.time()
	it = iter(ds)
	for i in range(steps):
		batch = next(it)
		if i % 10 == 0:
			print('.', end='')
	print()
	end = time.time()

	duration = end - start
	print("{} batches: {} s".format(steps, duration))
	print("{:0.5f} Images/s".format(BATCH_SIZE*steps/duration))

# timeit(train_data)

# model = tf.keras.models.Sequential([
# 	tf.keras.layers.
# 	tf.keras.layers.Flatten(input_shape=(IMG_WIDTH, IMG_HEIGHT)),
# 	tf.keras.layers.Dense(1024, activation='relu'),
# 	tf.keras.layers.Dense(len(CLASS_NAME), activation='softmax')
# ])

model = tf.keras.models.Sequential()
num_channels = 4
while num_channels < 4096:
	print(num_channels)
	model.add(tf.keras.layers.Conv2D(num_channels, kernel_size=3, padding='same'))
	model.add(tf.keras.layers.Conv2D(num_channels, kernel_size=5, padding='same'))
	model.add(tf.keras.layers.Activation('relu'))
	model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
	num_channels *= 4

print(num_channels)
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(1024, activation='relu'))
model.add(tf.keras.layers.Dense(len(CLASS_NAME), activation='softmax'))

print('Compiling')
model.compile(
	loss='sparse_categorical_crossentropy',
	optimizer=tf.keras.optimizers.Adam(0.001),
	metrics=['accuracy']
)

model.fit(
	train_data,
	epochs=20,
	validation_data=val_data
)

start = time.time()
model.save(str(start) + '.h5')