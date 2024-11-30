
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import model_from_json

import matplotlib.pyplot as plt
import warnings
import numpy as np
import os
import tensorflow as tf

from keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.preprocessing.image import ImageDataGenerator

#import torch
#from torch.utils.data import DataLoader
#from torchvision import datasets, transforms

# Define data paths
train_data_path = "./train"
batch_size = 32

# Data augmentation and normalization
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
# Create a generator for training data
train_generator = train_datagen.flow_from_directory(
        'train',  # source directory for training images
        target_size=(224, 224),  # All images are be resized to 200 x 200
        batch_size=batch_size,
        classes=['COVID19','NORMAL','PNEUMONIA'],
        class_mode='categorical'
)

# CNN Model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.summary()


# Updated Early stopping based on validation accuracy
class EarlyStoppingByAccuracy(tf.keras.callbacks.Callback):
    def __init__(self, monitor='accuracy', value=0.98, verbose=0):
        super(tf.keras.callbacks.Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs=None):
        current_accuracy = logs.get('accuracy')  # Change 'val_accuracy' to 'accuracy'
        if current_accuracy is None:
            warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if current_accuracy is not None and current_accuracy >= self.value:
            if self.verbose > 0:
                print("Epoch {}: Stopping training as accuracy reached {}".format(epoch + 1, self.value))
            self.model.stop_training = True

early_stopping = EarlyStoppingByAccuracy(monitor='accuracy', value=0.98, verbose=1)

model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
              metrics=['accuracy'])

total_sample = train_generator.n
n_epochs = 10

history = model.fit(
    train_generator,
    epochs=n_epochs,
    verbose=1,
    callbacks=[early_stopping]
)

model.save('model.keras')



acc = history.history['accuracy']

loss = history.history['loss']

epochs = range(1, len(acc) + 1)

# Train and validation accuracy
plt.plot(epochs, acc, 'b', label=' accurarcy')

plt.title('  accurarcy')
plt.legend()

plt.figure()

# Train and validation loss
plt.plot(epochs, loss, 'b', label=' loss')
plt.title('  loss')
plt.legend()
plt.show()
