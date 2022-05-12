import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import pickle

TRAIN_DIR = 'C:/Users/Kodee/Desktop/dogs-vs-cats-redux-kernels-edition/train/train'
TEST_DIR = 'C:/Users/Kodee/Desktop/dogs-vs-cats-redux-kernels-edition/test/test'
IMG_SIZE = 72

def label_img(img):
    word_label = img.split('.')[-3]
    if word_label == 'cat': 
        return 0
    elif word_label == 'dog': 
        return 1

def create_training_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_img(img)
        path = os.path.join(TRAIN_DIR, img)
        img = cv2.resize(cv2.imread(path, cv2.IMREAD_COLOR), (IMG_SIZE, IMG_SIZE))
        training_data.append([np.array(img), np.array(label)])
    shuffle(training_data)
    np.save('training_data.npy', training_data)
    return training_data

def process_testing_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR, img)
        img_num = img.split('.')[0]
        img = cv2.resize(cv2.imread(path, cv2.IMREAD_COLOR), (IMG_SIZE, IMG_SIZE))
        testing_data.append([np.array(img), img_num])
    np.save('test_data.npy', testing_data)
    return testing_data


training_data = create_training_data()

pickle_out = open('training_data.pickle', 'wb')
pickle.dump(training_data, pickle_out)
pickle_out.close()
