################ Henri Lahousse ################
# implementing image classification model
# 05/31/2022

# libraries
import cv2
import tensorflow as tf
import numpy as np

CATEGORIES = ["over", "under", "change_left", "change_right"] # possible categories

model = tf.keras.models.load_model("bw.model")    # trained model

img = r'ENTER_PATH' # image you want to predict

#cap = cv2.VideoCapture(0) # live video

#while True: # loop
  #  _, frame = cap.read()


# prepare image
def prepare(filepath):
    IMG_SIZE = 224  # image size needs to be same as in training the model!!
    img_array = cv2.imread(filepath)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    arr = new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3)
    return arr


x = prepare(img)
x = tf.keras.utils.normalize(x, axis=1)
x = tf.cast(x, tf.float32)  # change for right signature
# print(x.dtype)
prediction = model.predict([x])   # predicht class
# print(prediction)  # will be a list in a list.
print(CATEGORIES[np.argmax(prediction[0])]) # print predicted class
#res = np.argmax(prediction[0])  