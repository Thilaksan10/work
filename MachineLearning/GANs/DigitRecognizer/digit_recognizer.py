import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf
import csv


data = pd.read_csv('~/visualstudiocode-workspace/MachineLearning/GANs/DigitRecognizer/train.csv')
data.head()
training_data = np.array(data)

X_train = training_data[:, 1:]
y_train = training_data[:, 0]
X_train_norm = X_train / X_train.max()
X_train_norm = tf.reshape(X_train_norm, shape=[-1, 28, 28, 1])

data = pd.read_csv('~/visualstudiocode-workspace/MachineLearning/GANs/DigitRecognizer/test.csv')
data.head()
testing_data = np.array(data)
X_test_norm = testing_data / testing_data.max()
X_test_norm = tf.reshape(X_test_norm, shape=[-1, 28, 28, 1])

tf.random.set_seed(24)

model = tf.keras.Sequential(
    [
        tf.keras.layers.Conv2D(16, (3, 3), input_shape=X_train_norm.shape[1:], activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1)),
        tf.keras.layers.Conv2D(16, (3, 3), input_shape=X_train_norm.shape[1:], activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1)),
        tf.keras.layers.Conv2D(16, (3, 3), input_shape=X_train_norm.shape[1:], activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1)),
        tf.keras.layers.Conv2D(16, (3, 3), input_shape=X_train_norm.shape[1:], activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ]
)

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=tf.keras.optimizers.SGD(learning_rate=0.1), metrics=['accuracy'])

history = model.fit(X_train_norm, y_train, batch_size=32, epochs=25, validation_split=0.2)

pd.DataFrame(history.history).plot()
plt.xlabel('epochs')
plt.ylabel('loss')
plt.title('Digit Recognizer Loss Curve')
plt.show()

y_prob = model.predict(X_test_norm)
y_pred = y_prob.argmax(axis=1)

model.save('digit-recognizer')

with open('DigitRecognizer/submission.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['ImageId','Label'])
    for i in range(0, len(y_pred)):
        row = [i+1, y_pred[i]]
        writer.writerow(row)

    

