"""
Created on Thu Apr 8 09:47:00 2021

@author: dave-

describtion:
In this project, I used a pretrainied Xception model to classify the 
tensorflow flowers dataset.
Code was adapted from the book 'Praxiseinstieg Machine Learning mit Scikit-Learn,
Keras und Tensorflow' of A. Geron. chapter 14
    
Images used for testing this program were loaded via the
tensorflow function tfds.load.
Copyright used images: Copyright 2021 The TensorFlow Datasets Authors
"""


import matplotlib.pyplot as plt
import keras
import tensorflow as tf
import tensorflow_datasets as tfds

dataset, info = tfds.load("tf_flowers", as_supervised=True, with_info=True)
#with_info=true additional information about the dataset can be loaded 

dataset_size = info.splits["train"].num_examples # 3670
class_names = info.features["label"].names # ["dandelion", "daisy", ...]
n_classes = info.features["label"].num_classes # 5

print(dataset_size, class_names, n_classes)


test_split, valid_split, train_split = tfds.Split.TRAIN.subsplit([10, 15, 75])
test_set = tfds.load("tf_flowers", split=test_split, as_supervised=True)
valid_set = tfds.load("tf_flowers", split=valid_split, as_supervised=True)
train_set = tfds.load("tf_flowers", split=train_split, as_supervised=True)

###########################################################################
#preprocessing function for Xception


def preprocess(image, label):
    resized_image = tf.image.resize(image, [224, 224])
    final_image = keras.applications.xception.preprocess_input(resized_image)
    return final_image, label

 
batch_size = 32
train_set = train_set.shuffle(1000)
train_set = train_set.map(preprocess).batch(batch_size).prefetch(1)
valid_set = valid_set.map(preprocess).batch(batch_size).prefetch(1)
test_set = test_set.map(preprocess).batch(batch_size).prefetch(1)





###########################################################################
#loading of the pretrained Xception-model
base_model = keras.applications.xception.Xception(weights="imagenet",
                                                  include_top=False)
#include_top=False, to use own average pooling layer
#also use a different output dense layer


base_model = keras.applications.xception.Xception(weights="imagenet",
include_top=False)
avg = keras.layers.GlobalAveragePooling2D()(base_model.output)
#Mittelwert pooling
output = keras.layers.Dense(n_classes, activation="softmax")(avg)#input feld
#notation?
model = keras.Model(inputs=base_model.input, outputs=output)

optimizer = keras.optimizers.SGD(lr=0.2, momentum=0.9, decay=0.01)
model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer,
              metrics=["accuracy"])
history = model.fit(train_set, epochs=10, validation_data=valid_set, verbose=True)

print('Train_accuracy',history.history['accuracy'][-1])

plt.plot(history.history['loss'], color='b', label='Train')
plt.plot(history.history['val_loss'], color='r', label='Test')
plt.legend()
plt.show()



