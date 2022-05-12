import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

shape = [1000, 4]

minvalue = -100000
maxvalue = 100000

def generate_training_set(seed=0, shape=[1000, 100], range=(None, None)):
    generator = tf.random.Generator.from_seed(seed)
    x = generator.uniform(shape=shape, minval=range[0], maxval=range[1], dtype=tf.int32)
    return tf.random.shuffle(x, seed=seed)

X = generate_training_set(shape=shape, range=(minvalue, maxvalue))

print(X)

def sort_tensor(X=generate_training_set()):
    X_sorted = tf.sort(X).numpy()
    cols = X.shape[1]
    rows = X.shape[0]
    print(X_sorted)
    X_sorted_index = [[-1 for i in range(cols)] for j in range(rows)]
    for index1, x in enumerate(X):
        for index2, number2 in enumerate(x):
            for index3, number in enumerate(x):
                if number == X_sorted[index1][index2] and index3 + 1 not in X_sorted_index[index1]:
                    X_sorted_index[index1][index2] = index3 + 1
    return tf.convert_to_tensor(X_sorted_index)

y = sort_tensor(X=X)
print(y)

def prepare_data(X, y):
    X_normalized = tf.expand_dims(X/ np.array(X).max(), axis=-1)
    print(X_normalized.shape)
    y_normalized = y/ np.array(y).max()
    print(y_normalized.shape)
    return X_normalized, y_normalized

X_normalized, y_normalized = prepare_data(X,y)

model = tf.keras.models.Sequential(
    [
        tf.keras.layers.LSTM(1 * X_normalized.shape[1], batch_input_shape=(None, X_normalized.shape[1], X_normalized.shape[2]), return_sequences=True),
        tf.keras.layers.LSTM(8 * X_normalized.shape[1], batch_input_shape=(None, X_normalized.shape[1], X_normalized.shape[2]), return_sequences=True),
        tf.keras.layers.LSTM(8 * X_normalized.shape[1], batch_input_shape=(None, X_normalized.shape[1], X_normalized.shape[2]), return_sequences=True),
        tf.keras.layers.LSTM(1 * X_normalized.shape[1], return_sequences=False),
    ]
)

model.summary()

model.compile(loss=tf.keras.losses.MSE, optimizer=tf.keras.optimizers.Adam(), metrics=['msle', 'mae', 'accuracy'])

history = model.fit(x=X_normalized, y=y_normalized, batch_size=32, epochs=1000, validation_split=0.2)

pd.DataFrame(history.history).plot()
plt.xlabel('epochs')
plt.ylabel('loss')
plt.title('Loss Curve')

X_test = generate_training_set(seed=1, shape=[20, 4], range=(minvalue, maxvalue))
y_test = sort_tensor(X=X_test)

X_test_norm, y_test_norm = prepare_data(X=X_test, y=y_test)

for index, x in enumerate(X_test_norm):
    x = tf.expand_dims(x,axis=0)
    y_norm_predict = model.predict(x)
    y_round = np.array(tf.round(model.predict(x) * np.array(y).max()))
    print(f'Normalized IST: {y_norm_predict}')
    print(f'Normalized SOLL: {y_test_norm[index]}')
    print(f'IST: {y_round}')
    print(f'SOLL: {y_test[index]}')
    print(f'--------------------------------')

plt.show()



