import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import time

IMG_SIZE = 70

pickle_in = open('training_data.pickle', 'rb')
training_data = pickle.load(pickle_in)

X_train = np.array([i[0] for i in training_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y_train = np.array([i[1] for i in training_data])
X_train_norm = X_train / X_train.max()

dense_layers = [0, 1, 2]
layer_sizes = [32, 64, 128]
conv_layers = [1, 2, 3]

'''
for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            MODEL_NAME = '{}-conv-{}-nodes-{}-dense-{}'.format(conv_layer, layer_size, dense_layer, int(time.time()))
            print(MODEL_NAME)
            tensorboard = tf.keras.callbacks.TensorBoard(log_dir='logs/{}'.format(MODEL_NAME))

            model = tf.keras.Sequential(
                [
                    tf.keras.layers.Conv2D(layer_size, (3, 3), input_shape=X_train_norm.shape[1:], activation='relu'),
                    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1))
                ]
            )
            
            for l in range(conv_layer-1):
                model.add(tf.keras.layers.Conv2D(layer_size, (3, 3), activation='relu'))
                model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
            
            model.add(tf.keras.layers.Flatten())
            
            for l in range (dense_layer):
                model.add(tf.keras.layers.Dense(layer_size, activation='relu'))
            
            model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

            model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])

            history = model.fit(X_train_norm, y_train, batch_size=32, epochs=10, validation_split=0.3, callbacks=[tensorboard])

            pd.DataFrame(history.history).plot()
            plt.xlabel('epochs')
            plt.ylabel('loss')
            plt.title(MODEL_NAME)
            #plt.show()

            model.save(MODEL_NAME)
'''
conv_layer = 3
layer_size = 128
dense_layer = 0

MODEL_NAME = '{}-conv-{}-nodes-{}-dense-{}'.format(conv_layer, layer_size, dense_layer, int(time.time()))
print(MODEL_NAME)
tensorboard = tf.keras.callbacks.TensorBoard(log_dir='logs/{}'.format(MODEL_NAME))

model = tf.keras.Sequential(
    [
        tf.keras.layers.Conv2D(layer_size, (3, 3), input_shape=X_train_norm.shape[1:], activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1))
    ]
)

for l in range(conv_layer-1):
    model.add(tf.keras.layers.Conv2D(layer_size, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(tf.keras.layers.Flatten())

for l in range (dense_layer):
    model.add(tf.keras.layers.Dense(layer_size, activation='relu'))

model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])

history = model.fit(X_train_norm, y_train, batch_size=32, epochs=10, validation_split=0.3, callbacks=[tensorboard])

pd.DataFrame(history.history).plot()
plt.xlabel('epochs')
plt.ylabel('loss')
plt.title(MODEL_NAME)
#plt.show()

model.save(MODEL_NAME)
plt.show()

