import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

#loading dataset
fashion_mnist = keras.datasets.fashion_mnist 
#splitting data into training and testing 
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#taking a look at the data
print(train_images.shape)           
#taking a look at one pixel
print(train_images[0,23,23])
#looking at the first 10 labels
print(train_labels[:10])


#creating an array of label names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#plt.figure()
#plt.imshow(train_images[5])
#plt.colorbar()
#plt.grid(False)
#plt.show()

#preprocessing data
train_images = train_images / 255.0
test_images = test_images / 255.0


#Creating a model
model = keras.Sequential([
         keras.layers.Flatten(input_shape=(28, 28)), #Input layer
         keras.layers.Dense(128, activation='relu'), #Hidden layer
         keras.layers.Dense(10, activation='softmax') #Output layer
    ])


#Compiling the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#Training the model
model.fit(train_images, train_labels, epochs=14)

#Testing the model
#test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=1)

#print('Test accuracy:', test_acc)

predictions = model.predict([test_images])
print(class_names[np.argmax(predictions[5])])
plt.figure()
plt.imshow(test_images[5])
plt.colorbar()
plt.x_title(class_names[np.argmax(predictions[5])])
plt.grid(False)
plt.show()