# main.py
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
from sklearn.model_selection import train_test_split
import os
import pandas as pd

path = "/content/drive/MyDrive/Dataset"
labelFile = '/content/labels.csv'
batch_size_val = 32
epochs_val = 10
imageDimesions = (32, 32, 3)
channels = 1  # grayscale

testRatio = 0.2
validationRatio = 0.2

images = []
classNo = []
myList = os.listdir(path)
noOfClasses = len(myList)

for class_id in range(noOfClasses):
    class_path = os.path.join(path, str(class_id))
    myPicList = os.listdir(class_path)
    for img_name in myPicList:
        img_path = os.path.join(class_path, img_name)
        curImg = cv2.imread(img_path)
        if curImg is not None:
            curImg = cv2.resize(curImg, (imageDimesions[0], imageDimesions[1]))
            images.append(curImg)
            classNo.append(class_id)

images = np.array(images)
classNo = np.array(classNo)

X_train, X_test, y_train, y_test = train_test_split(images, classNo, test_size=testRatio)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validationRatio)

def grayscale(img): return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
def equalize(img): return cv2.equalizeHist(img)
def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255.0
    return img

X_train = np.array([preprocessing(img) for img in X_train])
X_validation = np.array([preprocessing(img) for img in X_validation])
X_test = np.array([preprocessing(img) for img in X_test])

X_train = X_train.reshape(-1, 32, 32, 1)
X_validation = X_validation.reshape(-1, 32, 32, 1)
X_test = X_test.reshape(-1, 32, 32, 1)

dataGen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2,
                             shear_range=0.1,
                             rotation_range=10)
dataGen.fit(X_train)

y_train = to_categorical(y_train, noOfClasses)
y_validation = to_categorical(y_validation, noOfClasses)
y_test = to_categorical(y_test, noOfClasses)

def myModel():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(noOfClasses, activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

model = myModel()
model.summary()
history = model.fit(dataGen.flow(X_train, y_train, batch_size=batch_size_val),
                    steps_per_epoch=len(X_train)//batch_size_val,
                    epochs=epochs_val,
                    validation_data=(X_validation, y_validation),
                    shuffle=True)

plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training', 'Validation'])
plt.title('Loss')
plt.xlabel('Epoch')

plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['Training', 'Validation'])
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.show()

score = model.evaluate(X_test, y_test, verbose=0)
print('Test Score:', score[0])
print('Test Accuracy:', score[1])

model.save("model.h5")
