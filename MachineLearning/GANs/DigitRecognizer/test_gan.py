import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

generator = tf.keras.models.load_model('generator')


rand = np.random.randn(1, 100)
generated = generator(rand)
generated_reshaped = tf.reshape(generated, (28, 28))
print(generated_reshaped)
plt.imshow(generated_reshaped, cmap='gray')
plt.show()

discriminator = tf.keras.models.load_model('discriminator')
decision = discriminator(tf.reshape(generated_reshaped, (-1, 28, 28, 1)))
print(decision)

digit_recognizer = tf.keras.models.load_model('digit-recognizer')
prediction = digit_recognizer(tf.reshape(generated_reshaped, (-1, 28, 28, 1)))
print(tf.argmax(prediction, axis=1))