# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images / 255.0
test_images = test_images / 255.0

input = tf.keras.layers.Input(shape=train_images.shape)
flattened = tf.keras.layers.Flatten(input_shape=(28,28))(input) 
hidden1 = tf.keras.layers.Dense(128, activation='softmax')(flattened)
hidden2 = tf.keras.layers.Dense(128, activation='relu')(hidden1) 
output = tf.keras.layers.Dense(10)(hidden2)
model = tf.keras.Model(inputs = [input], outputs = [output])
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.show()
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)