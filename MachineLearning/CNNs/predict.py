import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import pickle
import tensorflow as tf
import csv

TEST_DIR = 'C:/Users/Kodee/Desktop/dogs-vs-cats-redux-kernels-edition/test/test'
IMG_SIZE = 70

MODEL_NAME = '{}-conv-{}-nodes-{}-dense-{}'.format(3, 128, 0, 1634830529)


def process_testing_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR, img)
        img_num = img.split('.')[0]
        img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (IMG_SIZE, IMG_SIZE))
        testing_data.append([np.array(img), img_num])
    np.save('test_data.npy', testing_data)
    return testing_data

try:
    pickle_in = open('testing_data.pickle', 'rb')
    testing_data = pickle.load(pickle_in)
except:
    testing_data = process_testing_data()
    pickle_out = open('testing_data.pickle', 'wb')
    pickle.dump(testing_data, pickle_out)
    pickle_out.close()

print(len(testing_data))
print(testing_data[:1])
print(np.array(testing_data).shape)
#for i in testing_data[:5]:
#    print(i[0])
X_test = np.array([i[0] for i in testing_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
X_test_norm = X_test/X_test.max()

model = tf.keras.models.load_model(MODEL_NAME)

y_prob = model.predict(X_test_norm)
print(y_prob)

with open('submission.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['id','label'])
    for i in range(0, len(y_prob)):
        row = [i+1, y_prob[i, 0]]
        writer.writerow(row)