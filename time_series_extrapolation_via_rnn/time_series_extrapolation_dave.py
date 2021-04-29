# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 14:26:26 2021

@author: dave-

In this project, time series of cyclic functions were extraploated
via a recurrent neuronal network.
Used functions are included in the .py-document "time_series_funktion"

Code was adapted from the book 'Praxiseinstieg Machine Learning mit Scikit-Learn,
Keras und Tensorflow' of A. Geron. chapter 15, page 508-512


"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


#set the seed
np.random.seed(42)
tf.random.set_seed(42)



from time_series_funktion import generate_time_series
from time_series_funktion import plot_series
from time_series_funktion import save_fig

batch_size = 10000 #number of time series
n_steps = 50 #length time series

series = generate_time_series(batch_size, n_steps + 1)
X_train, y_train = series[:7000, :n_steps], series[:7000, -1]
X_valid, y_valid = series[7000:9000, :n_steps], series[7000:9000, -1]
X_test, y_test = series[9000:, :n_steps], series[9000:, -1]

print(X_train.shape, X_valid.shape, X_test.shape)
# Here, the goal is to predict 1 value for each time series:
    
    
fig, axes = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(12, 4))
for col in range(3):
    plt.sca(axes[col])
    plot_series(X_valid[col, :, 0], y_valid[col, 0],
                y_label=("$x(t)$" if col==0 else None))
#save_fig("time_series_plot")
plt.show()


# RNN (3 layers)

model = keras.models.Sequential([
    keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
    #activation function is tanh
    #return_sequnces=True gives the result for each time step,
    #which are needed for the extrapolation
    keras.layers.SimpleRNN(20),
    #keras.layers.SimpleRNN(1)
    #alternatively to the third layer, a dense layer is possible:
    keras.layers.Dense(1)
])

model.compile(loss="mse", optimizer="adam")
history = model.fit(X_train, y_train, epochs=20,
                    validation_data=(X_valid, y_valid))


print(model.evaluate(X_valid, y_valid))


from time_series_funktion import plot_learning_curves

#learning curves are plotted via the loss und val_loss values
plot_learning_curves(history.history["loss"], history.history["val_loss"])
plt.show()


y_pred = model.predict(X_valid)
fig, axes = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(12, 4))
for col in range(3):
    plt.sca(axes[col])
    plot_series(X_valid[col, :, 0], y_valid[col, 0], y_pred[col, 0],
                y_label=("$x(t)$" if col==0 else None))
#save_fig("time_series_plot")
plt.show()










