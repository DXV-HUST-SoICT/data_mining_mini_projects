import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
import time

tf.enable_v2_behavior()

(ds_train, ds_test), ds_info = tfds.load(
	'mnist', split=['train', 'test'],
	shuffle_files=True,
	as_supervised=True,
	with_info=True
)

print(type(ds_train))

def normalize_img(image, label):
	return tf.cast(image, tf.float32) / 255., label

ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(128)
ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

# print(type(ds_train))
# for images, labels in ds_train.take(1):
# 	images_numpy = images.numpy()
# 	labels_numpy = labels.numpy()
# 	print(images_numpy)
# 	print(labels_numpy)
# 	break

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Conv2D(20, kernel_size=3, input_shape=(28, 28, 1)))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(tf.keras.layers.Conv2D(50, kernel_size=3))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(tf.keras.layers.Flatten(input_shape=(28, 28, 1)))
model.add(tf.keras.layers.Dense(512))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.Dense(10))
model.add(tf.keras.layers.Activation('softmax'))

print('Compiling')
model.compile(
	loss='sparse_categorical_crossentropy',
	# optimizer=tf.keras.optimizers.SGD(),
	optimizer=tf.keras.optimizers.Adam(0.001),
	metrics=['accuracy']
)

model.fit(
	ds_train,
	epochs=20,
	validation_data=ds_test
)

start = time.time()
model.save(str(start) + '.h5')