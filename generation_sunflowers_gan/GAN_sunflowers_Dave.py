# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 10:59:00 2021

@author: dave-

describtion:
With this small programm, you can generate images
via a generative adversarial network (GAN).
Find some samples of generated sunflowers in the attached folder
"samples_of_generated_images"
When running the code, be patient, depending on your system, 
this will take a while ;)


Images used for testing this program are from: https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/image_classification/flowers.py
For my example, I used the sunflowers only.
Copyright used images: Copyright 2021 The TensorFlow Datasets Authors
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import PIL
import PIL.Image
import keras


#Link of your image folder
data_dir = r'link_image_folder'



batch_size = 699
img_height = 112
img_width = img_height


train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir, 
  image_size=(img_height, img_width),# distortion of images possible
  batch_size=batch_size)

class_names = train_ds.class_names

#####################################################################

    
print(class_names)



plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")
    
############################################################
#preprocessing:

    
real_data = train_ds
real_data.prefetch(buffer_size=-1)
X_train=(list(real_data.as_numpy_iterator())[0])

#label separation:
y_labels = X_train[1]
X_train = X_train[0]


X_train = X_train.reshape(-1, 112, 112, 3)/255 * 2. - 1. # Reshape und Rescale

print(X_train.shape, type(X_train))


###########################################################
#model:
codings_size = 100
generator = keras.models.Sequential([
keras.layers.Dense(7 * 7 * 128, input_shape=[codings_size]),
keras.layers.Reshape([7, 7, 128]),
keras.layers.BatchNormalization(),
keras.layers.Conv2DTranspose(64, kernel_size=5, strides=2, padding="same",
activation="selu"),
keras.layers.BatchNormalization(),
keras.layers.Conv2DTranspose(32, kernel_size=5, strides=2, padding="same",
activation="selu"),
keras.layers.BatchNormalization(),
keras.layers.Conv2DTranspose(16, kernel_size=5, strides=2, padding="same",
activation="selu"),
keras.layers.BatchNormalization(),
keras.layers.Conv2DTranspose(3, kernel_size=5, strides=2, padding="same",
activation="tanh")
])

discriminator = keras.models.Sequential([
keras.layers.Conv2D(64, kernel_size=5, strides=2, padding="same",
activation=keras.layers.LeakyReLU(0.2),
input_shape=[112, 112, 3]),
keras.layers.Dropout(0.4),
keras.layers.Conv2D(128, kernel_size=5, strides=2, padding="same",
activation=keras.layers.LeakyReLU(0.2)),
keras.layers.Dropout(0.4),
keras.layers.Conv2D(64, kernel_size=5, strides=2, padding="same",
activation=keras.layers.LeakyReLU(0.2)),
keras.layers.Dropout(0.4),
keras.layers.Conv2D(32, kernel_size=5, strides=2, padding="same",                    
activation=keras.layers.LeakyReLU(0.2)),
keras.layers.Dropout(0.4),
keras.layers.Flatten(),
keras.layers.Dense(1, activation="sigmoid")
])

gan = keras.models.Sequential([generator, discriminator])

discriminator.compile(loss="binary_crossentropy", optimizer="rmsprop")
discriminator.trainable = False
gan.compile(loss="binary_crossentropy", optimizer="rmsprop")

batch_size = 32
dataset = tf.data.Dataset.from_tensor_slices(X_train).shuffle(1000)
dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(1)

def train_gan(gan, dataset, batch_size, codings_size, n_epochs=5000):
    generator, discriminator = gan.layers
    for epoch in range(n_epochs):
                print('---- Epoch: %i ----'%epoch)
                for X_batch in dataset:
                    # phase 1 - train discriminator
                    noise = tf.random.normal(shape=[batch_size, codings_size])
                    generated_images = generator(noise)
                    X_fake_and_real = tf.concat([generated_images, X_batch], axis=0)
                    y1 = tf.constant([[0.]] * batch_size + [[1.]] * batch_size)
                    discriminator.trainable = True
                    discriminator.train_on_batch(X_fake_and_real, y1)
                    # phase 2 - train generator
                    noise = tf.random.normal(shape=[batch_size, codings_size])
                    y2 = tf.constant([[1.]] * batch_size)
                    discriminator.trainable = False
                    gan.fit(noise, y2, verbose = 1)
                plt.imshow(generated_images[0], interpolation='nearest', cmap='gray_r')
                plt.show()
                    
train_gan(gan, dataset, batch_size, codings_size)













