import cv2
import csv
import numpy as np

lines = []
#with open('data/data/driving_log.csv') as csvfile:
with open('/home/carnd/data/CarND-Behavioral-Cloning-P3/data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

images = []
measurements = []
for line in lines[1:]:
	directory = '/home/carnd/data/CarND-Behavioral-Cloning-P3/data/IMG/'
	filenames = [t.split('/')[-1] for t in line[0:3]]
	img_center = cv2.imread(directory + filenames[0])
	img_left = cv2.imread(directory + filenames[1])
	img_right = cv2.imread(directory + filenames[2])

	#create adjusted steering adjustments for the side camera images
	correction = 0.2 #parameter
	steering_center = float(line[3])
	steering_left = steering_center + correction
	steering_right = steering_center - correction

	#add images and angles to data set
	images.extend([img_center, img_left, img_right])
	measurements.extend([steering_center, steering_left, steering_right])
	images.extend([np.fliplr(img_center), np.fliplr(img_left), np.fliplr(img_right)])
	measurements.extend([-steering_center, -steering_left, -steering_right])

X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D

model = Sequential()
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160, 320, 3)))
model.add(Lambda(lambda x: x / 255.0 - 0.5))
model.add(Convolution2D(24, 5, 5, activation='relu', strides=(2, 2)))
model.add(Convolution2D(36, 5, 5, activation='relu', strides=(2, 2)))
model.add(Convolution2D(48, 5, 5, activation='relu', strides=(2, 2)))
model.add(Convolution2D(64, 3, 3, activation='relu', strides=(1, 1)))
model.add(Convolution2D(64, 3, 3, activation='relu', strides=(1, 1)))
model.add(Flatten())
#print(model.summary())

model.add(Dense(8448))
model.add(Dense(850))
model.add(Dense(450))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss = 'mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)

model.save('model.h5')
