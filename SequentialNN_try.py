
import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images / 255.0
test_images = test_images /255.0

model = keras.Sequential( #NN type = "Sequential"
                          [keras.layers.Flatten(input_shape=(28,28)), #input layer 0 (784 N) Flatten - flatten matrix)))
                          keras.layers.Dense(128, activation='relu'), #hidden layer 1 (128 N)
                          keras.layers.Dense(10, activation='softmax') # output layer 2 (10 N)
                          ])

model.compile(optimizer='adam',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

model.fit(train_images, train_labels, epochs=1)

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=1)
print('Test accuracy: ', test_acc)

while True:
    img_ind = input('Enter image index: ')
    predictions = model.predict([test_images])
    print(class_names[np.argmax(predictions[int(img_ind)])])
    plt.figure()
    plt.imshow(test_images[int(img_ind)])
    plt.colorbar()
    plt.grid(False)
    plt.show()