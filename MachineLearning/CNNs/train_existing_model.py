import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pickle
import pandas as pd

IMG_SIZE = 28
learning_rate = 1e-1

MODEL_NAME = 'dogsvscats-{}-{}.model'.format(learning_rate, '8conv16-4x4-4dense-sgd')

pickle_in = open('training_data.pickle', 'rb')
training_data = pickle.load(pickle_in)

train = training_data[:-1000]
test = training_data[-1000:]

X_train = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
y_train = np.array([i[1] for i in train])
X_train_norm = X_train / X_train.max()
print(X_train.max())

X_test = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
y_test = np.array([i[1] for i in test])
X_test_norm = X_test / X_test.max()
print(X_test.max())

model = tf.keras.models.load_model(MODEL_NAME)

#model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate), metrics=['accuracy'])

history = model.fit(X_train_norm, y_train, batch_size=32, epochs=25, validation_data=(X_test_norm, y_test))

pd.DataFrame(history.history).plot()
plt.xlabel('epochs')
plt.ylabel('loss')
plt.title('Dog vs Cat loss curve')
plt.show()

model.save(MODEL_NAME)