from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import numpy as np


"""
Train the CNN using dataset
"""

# load saved faces and labels
data = np.load('data.npy')
target = np.load('target.npy')

model = Sequential()

# adding a convolution layer of 200 filters of size 3x3
model.add(Conv2D(200, (3, 3), input_shape=data.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# adding a convolution layer of 100 filters of size 3x3
model.add(Conv2D(100, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten()) # add a flatten layer
model.add(Dropout(0.5))  # add a dropout layer to reduce over-fitting
model.add(Dense(50, activation='relu')) # add a dense layer

# final layer with two outputs for two categories
model.add(Dense(2, activation='softmax'))

# metrics['accuracy'] will print the accuracy
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 10% of test data
train_data,test_data,train_target,test_target = train_test_split(data, target, test_size=0.1)

# monitor validation loss and save best model per each epoch
# if the validation loss get increased after epoch, it will not be saved
checkpoint = ModelCheckpoint('models/model-{epoch:03d}.model', monitor='val_loss', verbose=0, save_best_only=True, mode='auto')

# use 20% of validation data
history = model.fit(train_data, train_target, epochs=20, callbacks=[checkpoint], validation_split=0.2)

# plotting loss
plt.plot(history.history['loss'], 'r', label='training loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.xlabel('Number of epochs')
plt.ylabel('loss')
plt.legend()
plt.savefig('loss.png')
plt.show()

# plotting accuracy
plt.plot(history.history['accuracy'], 'r', label='training accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.xlabel('Number of epochs')
plt.ylabel('accuracy')
plt.legend()
plt.savefig('accuracy.png')
plt.show()

# print loss and testing accuracy
print(model.evaluate(test_data, test_target))

