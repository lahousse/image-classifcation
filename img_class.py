################ Henri Lahousse ################
# training image classification model
# not originaly made by me, I just adjusted some stuff and made it up to date. 
# 05/31/2022

# libraries

# visualisation
import matplotlib.pyplot as plt
import seaborn as sns

# models
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

from sklearn.metrics import classification_report, confusion_matrix

# data conversion
import cv2
import os
import numpy as np


labels = ['change_left', 'change_right', 'over', 'under']  # adjust possible labels, these are the classes you want to categories
img_size = 224                                             # image size

def get_data(data_dir):
    data = []
    for label in labels:
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img))[..., ::-1]  # BGR naar RGB format
                resized_arr = cv2.resize(img_arr, (img_size, img_size))   # reshaping images
                data.append([resized_arr, class_num])                     
            except Exception as e:
                print(e)
    return np.array(data)

# train and test different data!!!!! testing on seen data is not accurate representation
train = get_data(r'ENTER_PATH')  # training set, at least 60% of data
val = get_data(r'ENTER_PATH')     # test set

# representing amounts of each class in train folder
l = []
for i in train:

    if (i[1] == 1):
        l.append("change_left")
    elif (i[1] == 1):
        l.append("change_right")
    elif (i[1] == 0):
        l.append("over")
    elif (i[1] == 1):
        l.append("under")

# show in histogram with seaborn
sns.set_style('darkgrid')
sns.countplot(l)

plt.figure(figsize=(5, 5))
plt.imshow(train[1][0])
plt.title(labels[train[0][1]])

x_train = []
y_train = []
x_val = []
y_val = []

# appending train list with train data
for feature, label in train:
  x_train.append(feature)
  y_train.append(label)

# appending validation list with testing data
for feature, label in val:
  x_val.append(feature)
  y_val.append(label)

# normalize data
x_train = np.array(x_train) / 255
x_val = np.array(x_val) / 255

x_train.reshape(-1, img_size, img_size, 1)
y_train = np.array(y_train)

x_val.reshape(-1, img_size, img_size, 1)
y_val = np.array(y_val)

# for generating more data, BE VERY CAREFULLY HOW YOU GENERATE DATA!! 
# Change this according to your application, ex: don't flip images if you want to train something based on the angle of the image.
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range = 30,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.2, # Randomly zoom image
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip = True,  # randomly flip images
        vertical_flip=False # randomly flip images
    )

# train data
# layers
datagen.fit(x_train)

model = Sequential()
model.add(Conv2D(32, 3, padding="same", activation="relu", input_shape=(224, 224, 3)))
model.add(MaxPool2D())

model.add(Conv2D(32, 3, padding="same", activation="relu"))
model.add(MaxPool2D())

model.add(Conv2D(64, 3, padding="same", activation="relu"))
model.add(MaxPool2D())
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dense(4, activation="softmax"))                                          # aanpassen voor aantal klasses

model.summary()

# optimization
opt = Adam(learning_rate=0.000001)
model.compile(optimizer=opt, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=500, validation_data=(x_val, y_val))

# represent train data
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(500)  # aantal keer trainen

# show data in histogram
plt.figure(figsize=(15, 15))
plt.subplot(2, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# prediction and testing how accurate output
predict_x = model.predict(x_val)
predictions = np.argmax(predict_x, axis=1)

print(classification_report(y_val, predictions, target_names=['change_left', 'change_right', 'over', 'under']))

# save model, to use implement model in the img_prod code 
model.save('bw.model')