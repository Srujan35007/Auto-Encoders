import time
bef = time.time()
import matplotlib.pyplot as plt
import numpy as np
from tqdm.notebook import tqdm
import math
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout, BatchNormalization, MaxPool2D
from tensorflow.keras import models, datasets
aft = time.time()
print(f'Imports complete in {aft-bef} seconds')

# Encoder
encoder = models.Sequential()
encoder.add(Conv2D(32, kernel_size=(2, 2),
                   activation='sigmoid', input_shape = (28,28,1)))
encoder.add(Conv2D(64, kernel_size=(2, 2), activation='sigmoid'))
encoder.add(MaxPool2D((2,2)))
encoder.add(Dropout(0.3))
encoder.add(Flatten())
encoder.add(Dense(100, activation='relu'))
encoder.add(BatchNormalization())
encoder.add(Dense(100, activation='relu'))
encoder.add(Dropout(0.3))
encoder.add(Dense(10, activation='softmax'))
print(f'Encoder created.')
encoder.compile(loss = 'sparse_categorical_crossentropy',\
     optimizer='adam', metrics=['accuracy'])
print('Encoder compiled.')

# Decoder
decoder = models.Sequential()
decoder.add(Dense(100, activation='relu'))
decoder.add(Dense(100, activation='relu'))
decoder.add(BatchNormalization())


#--------------fitting and testing------------------
(X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()
X_train = X_train/255
X_test = X_test/255
X_train = X_train.reshape(60000,28,28,1)
X_test = X_test.reshape(10000,28,28,1)
print('Data ready')
encoder.fit(X_train, y_train, epochs = 7,\
     batch_size=40, validation_data=(X_test, y_test))
