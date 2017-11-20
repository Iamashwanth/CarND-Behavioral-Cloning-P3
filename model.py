import csv
import cv2
import sklearn
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from random import randint
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D

samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

samples.pop(0)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    correction_factor = [0.2, 0.0, -0.2]

    while 1:
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset: offset+batch_size]

            images = []
            steering = []
            for batch_sample in batch_samples:
                camera = randint(0, len(correction_factor)-1)
                filename = batch_sample[camera].split('/')[-1]
                current_path = './data/IMG/' + filename
                image = mpimg.imread(current_path)
                measurement = float(batch_sample[3]) + correction_factor[camera]
                
                flip = randint(0, 1)    
                if flip:
                    images.append(cv2.flip(image, 1))
                    steering.append(measurement * -1.0)
                else:
                    images.append(image)
                    steering.append(measurement)
                        
            X_train = np.array(images)
            y_train = np.array(steering)
            yield sklearn.utils.shuffle(X_train, y_train)

train_generator = generator(train_samples)
validation_generator = generator(validation_samples)

model = Sequential()
model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape = (160, 320, 3)))
model.add(Lambda(lambda x: x / 128.0 - 1.0))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation = 'relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation = 'relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation = 'relu'))
model.add(Convolution2D(64, 3, 3, activation = 'relu'))
model.add(Convolution2D(64, 3, 3, activation = 'relu'))
model.add(Flatten())
model.add(Dense(100, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(50, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
history_obj = model.fit_generator(train_generator, samples_per_epoch =\
            len(train_samples), validation_data = validation_generator,\
            nb_val_samples = len(validation_samples), nb_epoch = 3, verbose = 1)

print(history_obj.history.keys())

#plt.plot(history_obj.history['loss'])
#plt.plot(history_obj.history['val_loss'])
#plt.ylabel('mean squared error loss')
#plt.xlabel('epoch')
#plt.legend(['training set', 'validation set'], loc='upper right')
#plt.show()

model.save('model.h5')
