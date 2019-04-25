import numpy as np
import os
import cv2
import random
import pickle

# enter the absolute path to the directory where the dataset is stored after 'DATADIR = '
# the directory should contain 2 directories: one called 'free' and one called 'busy'
# for more info on this check the Handleiding
DATADIR = ""
CATEGORIES = ["free", "busy"]


# this function creates the training data by changing every image to greyscale, by changing the size of the image and
# by adding every image to the list training_data
def create_training_data():
    training_data = []
    IMG_SIZE = 50
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass

    # the data gets shuffled so it's not in a certain order
    random.shuffle(training_data)

    # the features and the labels of the images get saved in an X and y variable
    X = []
    y = []
    for features, label in training_data:
        X.append(features)
        y.append(label)

    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

    # X and y are saved with pickle so they can be used in a different python file
    pickle_out = open("X.pickle", "wb")
    pickle.dump(X, pickle_out)
    pickle_out.close()

    pickle_out = open("y.pickle", "wb")
    pickle.dump(y, pickle_out)
    pickle_out.close()

create_training_data()