import tensorflow as tf
import numpy as np

x = tf.constant(1, shape=[32, 50, 50, 1])

tf.reshape(x, [-1, 50, 50, 1])
y = tf.constant([1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1])

print(x)
conv = tf.keras.layers.Conv2D(32, (4,4), input_shape=x.shape[1:], activation='relu')

conv2 = tf.keras.layers.Conv2D(32, (4,4), activation='relu')

max_pool_2d = tf.keras.layers.MaxPooling2D(pool_size=(2, 2),strides=(1, 1), padding='valid')

model = tf.keras.Sequential(
    [
        conv,
        max_pool_2d,
        conv2,
        max_pool_2d,
        conv2,
        max_pool_2d,
        conv2,
        max_pool_2d,
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ]
)

model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.SGD(), metrics=['accuracy'])

model.fit(x, y, epochs=50)