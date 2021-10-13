from keras.models import load_model
import cv2
import numpy as np

"""
Detect the faces and masks using the video feed
"""

# observe the "models" folder and add a suitable model here
# higher number of models means higher accuracy
model = load_model('models/model-018.model')

# load the cascade classifier
cascade_classifier = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_default.xml')

# capture from the videofeed/web camera
source = cv2.VideoCapture(0)

# mapping for the labels to display
labels_map = {0: 'NO MASK', 1: 'MASK'}
# green for mask, red for no mask
color_map = {0: (0,0,255), 1: (0,255,0)}

while True:
    ret, img = source.read() # read from the videofeed
    img = cv2.flip(img, flipCode=1)  # flip the image for diplay purposes
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert to grayscale
    # detect faces using the cascade classifier
    faces = cascade_classifier.detectMultiScale(gray, 1.3, 5)

    # iterate for each face in the image
    for x, y, w, h in faces:
        face_img = gray[y:y+h, x:x+w] # detect the face region
        resized = cv2.resize(face_img, (100, 100)) # resize to 100x100

        # normalize into range 0-1
        normalized = resized/255.0 
        # convert to 4D for neural network
        reshaped = np.reshape(normalized, (1,100,100,1))

        # predict using the model
        result = model.predict(reshaped)
        print(result)

        # Returns the indices of the maximum values along an axis.
        label = np.argmax(result, axis=1)[0]

        # draw colord rectangles around the face
        cv2.rectangle(img, (x,y), (x+w,y+h), color_map[label],  2)
        cv2.rectangle(img, (x,y-40), (x+w,y), color_map[label], -1) 
        # put a white label for the face
        cv2.putText(img, labels_map[label], (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255))

    # show the output
    cv2.imshow('VideoFeed', img)
    key = cv2.waitKey(1)

    # quit when press Escape
    if key == 27:
        break

cv2.destroyAllWindows()
source.release()
