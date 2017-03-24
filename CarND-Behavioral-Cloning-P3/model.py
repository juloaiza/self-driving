
# Load pickled data
import  pprint, csv
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import sklearn

# TODO: Open csv file
def open_csv(path):
    rows = []
    with open(path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
	    #next(csvreader)
        for row in csvreader:
            rows.append(row)
    print("Total size per track ",len(rows))
    return rows

# TODO: Data Augmentation
def augmentation(driving_log_row):
    path ="" # "trainning/"
    data_augmented = []

    #Original
    image_angle = {}
    image_path = path + driving_log_row[0]
    image_angle["image"] = mpimg.imread(image_path)
    image_angle["angle"]= float(driving_log_row[3])
    data_augmented.append(image_angle)

    #Flipped
    image_angle = {}
    image_path = path + driving_log_row[0]
    image_angle["image"] = np.fliplr(mpimg.imread(image_path))
    image_angle["angle"]= -float(driving_log_row[3])
    data_augmented.append(image_angle)

    #Sides Camera
    correction = 0.2

    #Left Camera
    image_angle = {}
    image_path = path + driving_log_row[1].strip()
    image_angle["image"] = mpimg.imread(image_path)
    image_angle["angle"]= float(driving_log_row[3]) + correction
    data_augmented.append(image_angle)

    #Right Camera
    image_angle = {}
    image_path = path + driving_log_row[2].strip()
    image_angle["image"] = mpimg.imread(image_path)
    image_angle["angle"]= float(driving_log_row[3]) - correction
    data_augmented.append(image_angle)

    return data_augmented

# TODO:Generator
from sklearn.utils import shuffle
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                #name = 'trainning/IMG/'+batch_sample[0].split('/')[-1]
                data_augmented = augmentation(batch_sample)

                for row in data_augmented: 
                    images.append(row['image'])
                    angles.append(row['angle'])

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# Initial Setup for Keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Cropping2D
from keras.layers import Lambda

# TODO:Building Convolutional Neural Network in Keras
def architecture():
	model = Sequential()
	model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))  #Feature Scaling (Normalizing)
	model.add(Cropping2D(cropping=((70,25), (0,0)))) #70 row pix from top & 25 rows from bottom
	model.add(Convolution2D(36, 5, 5,subsample=(2,2)))
	model.add(Activation("relu"))
	model.add(Convolution2D(48, 5, 5,subsample=(2,2)))
	model.add(Activation("relu"))
	model.add(Convolution2D(64, 3, 3))
	model.add(Activation("relu"))
	model.add(Convolution2D(64, 3, 3))
	model.add(Activation("relu"))
	model.add(Flatten())
	model.add(Dense(100))
	model.add(Dropout(0.5)) #dropout 50%
	model.add(Dense(50))
	model.add(Dropout(0.5)) #dropout 50%
	model.add(Dense(10))
	model.add(Dropout(0.5)) #dropout 50%
	model.add(Dense(1))
	return model


### Main
driving_logs = [] 

path = "trainning/track1/"
driving_logs.append(open_csv(path + 'driving_log.csv'))

path = "trainning/track2/"
driving_logs.append(open_csv(path + 'driving_log.csv'))

driving_logs = driving_logs[0] + driving_logs[1]

print("Total size ",len(driving_logs))

# TODO: Split data into training and validation sets.
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(driving_logs, train_size=0.8)

# TODO: Train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

model = architecture()
model.compile(optimizer = 'adam', loss = 'mse')
history_object = model.fit_generator(train_generator, samples_per_epoch= 
    4*len(train_samples), validation_data=validation_generator, 
    nb_val_samples=4*len(validation_samples), nb_epoch=5,verbose = 1)

model.save('model.h5') #Model save

### print the keys contained in the history object
print(history_object.history.keys())

import matplotlib.pyplot as plt
### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
