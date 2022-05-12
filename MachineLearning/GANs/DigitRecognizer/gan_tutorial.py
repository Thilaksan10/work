import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
import time

from tqdm import tqdm
from IPython import display

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()


print(train_images.shape)

train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5)/ 127.5

BUFFER_SIZE = train_images.shape[0]
BATCH_SIZE = 32
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype('float32')
test_images = (test_images - 127.5)/ 127.5

'''
BUFFER_SIZE = 1000
BATCH_SIZE = 256
test_dataset = tf.data.Dataset.from_tensor_slices(test_images[:1000]).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
'''

# Generator
def make_generator_model():
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(7*7*256, use_bias=False, input_shape=(100,)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),

            tf.keras.layers.Reshape((7, 7, 256)),

            tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),

            tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),

            tf.keras.layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')
        ]
    )
    return model

generator = make_generator_model()

noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)

#plt.imshow(generated_image[0, :, :, 0], cmap='gray')
#plt.show()

# Discriminator
def make_discriminator_model():
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=train_images.shape[1:], activation='leaky_relu'),
            tf.keras.layers.Dropout(0.3),

            tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same', activation='leaky_relu'),
            tf.keras.layers.Dropout(0.3),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1)
        ]
    )
    return model

discriminator = make_discriminator_model()

decision = discriminator(generated_image)
print (decision)

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def get_discriminator_loss(real_predictions, fake_predictions):
    real_loss = cross_entropy(tf.ones_like(real_predictions), real_predictions)
    fake_loss = cross_entropy(tf.zeros_like(fake_predictions), fake_predictions)
    return fake_loss + real_loss

def get_generator_loss(fake_predictions):
    fake_loss = cross_entropy(tf.ones_like(fake_predictions), fake_predictions)
    return fake_loss

discriminator_optimizer = tf.optimizers.Adam(1e-4)
generator_optimizer = tf.optimizers.Adam(1e-4)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

# Training

EPOCHS = 50
noise_dim = 100
num_examples_to_generate = 16

# You will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise_dim])
print(seed)
print(seed.shape)

@tf.function
def train_step(images):
    fake_image_noise = tf.random.normal([BATCH_SIZE, noise_dim])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(fake_image_noise, training=True)
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = get_generator_loss(fake_output)
        disc_loss = get_discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return tf.math.reduce_mean(gen_loss), tf.math.reduce_mean(disc_loss)

def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()
        gen_loss = []
        disc_loss = []
        print('Epoch: {}/{}'.format(epoch+1,epochs))
        for image_batch in tqdm(dataset):
            mean_gen_loss, mean_disc_loss = train_step(image_batch)
            gen_loss.append(mean_gen_loss)
            disc_loss.append(mean_disc_loss)
        
        # Produce images for the GIF as you go
        display.clear_output(wait=True)
        generate_and_save_images(generator, epoch + 1, seed)

        # Save the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

        print('Generator Loss: ', np.mean(np.array(gen_loss)))
        print('Discriminator Loss: ', np.mean(np.array(disc_loss)))
        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
    
    # Generate after the final epoch
    display.clear_output(wait=True)
    generate_and_save_images(generator, epochs, seed)

def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    #plt.show()

train(train_dataset, EPOCHS)
plt.show()

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

def display_image(epoch_no):
    return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))

display_image(EPOCHS)

anim_file = 'dcgan.gif'

with imageio.get_writer(anim_file, mode='I') as writer:
    filenames = glob.glob('image*.png')
    filenames = sorted(filenames)
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
    image = imageio.imread(filename)
    writer.append_data(image)

import tensorflow_docs.vis.embed as embed
embed.embed_file(anim_file)

generator.save('generator')
discriminator.save('discriminator')

'''
[6],[3],[0],[6],
[9],[9],[2],[1],
[9],[4],[3],[0],
[3],[8],[5],[5]
'''