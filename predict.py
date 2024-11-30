
import os
import warnings
import tensorflow as tf
from keras.preprocessing import image
import numpy as np

classes = ["COVID19","NORMAL", "PNEUMONIA"]

# Load the trained model
classifierLoad = tf.keras.models.load_model('model.h5')

# Load the test image
img_path = './train/NORMAL/NORMAL(0).JPG'
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0  # Normalize the pixel values

# Predict the class
result = classifierLoad.predict(img_array)
print(result)

# Get the predicted class
predicted_class = np.argmax(result)

# Check the predicted class and print the corresponding class name
if predicted_class == 0:
    out = classes[0]
elif predicted_class == 1:
    out = classes[1]
elif predicted_class == 2:
    out = classes[2]
else:
    print("Unknown class")

print(out)
