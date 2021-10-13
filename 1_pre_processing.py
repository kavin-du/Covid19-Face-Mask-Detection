import cv2
import numpy as np
import os
from keras.utils import np_utils

"""
dataset pre-processing code
(extracting the ROI)

"""

# dataset should be avaible in this path before running the code
data_path = 'dataset/v2'
categories = os.listdir(data_path)
# adding a numerical value for each category
labels = [i for i in range(len(categories))]

# creating a mapping for category and label
label_map = dict(zip(categories, labels))
cascade_classifier = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_default.xml')

size = 100  # size of the image
data = []
target = []  # target label for the data

# traverse each folder
for category in categories:
    folder_path = os.path.join(data_path, category)
    img_names = os.listdir(folder_path)

    # traverse each image
    for img_name in img_names:
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path) # read the image
        
        try:
            # convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # detect the faces in the image
            faces = cascade_classifier.detectMultiScale(gray, 1.3, 5)

            # iterating each face in the image
            for x, y, w, h in faces:
                face_img = gray[y:y+h, x:x+w]  # extract the face area
                resized = cv2.resize(face_img, (size, size))
                
                # save each face to disk for testing purposes
                # if not os.path.exists(os.path.join('temp', category)):
                #     os.makedirs(os.path.join('temp', category))
                
                # file_name = f'temp/{category}/{img_name}'
                # cv2.imwrite(file_name, resized)
                # print(file_name)

                data.append(resized) # add face to the array
                target.append(label_map[category]) # add label for the face

        except Exception as e:
            print('Exception raised:', e)

# normalize the images for 0-1 range
data = np.array(data) / 255.0 
# convert to 4D array, since neural network accept 4D arrays
data = np.reshape(data, (data.shape[0], size, size, 1))
target = np.array(target) # convert labels to numpy array

new_target = np_utils.to_categorical(target)

# delete the files if already exist
if os.path.exists('data.npy'):
    os.remove('data.npy')
if os.path.exists('target.npy'):
    os.remove('target.npy')   
          
np.save('data', data) # save the faces to disk
np.save('target', new_target) # save the labels to disk
