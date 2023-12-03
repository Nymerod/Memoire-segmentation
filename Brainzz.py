#L3 MIASHS Mémoire 

import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
from PIL import Image
import cv2

the_seed = np.random.seed(1000)

########### image

#test_img et valid_img sont pour l'instant le même test dataset, l'ia traitera avec un validation dataset une fois résultat OK sur training dataset

test_img = tf.keras.preprocessing.image_dataset_from_directory(
    directory="j",
    labels="inferred",
    label_mode="binary",
    color_mode="grayscale",
    batch_size=32,
    shuffle=True,
    seed=the_seed,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
)
valid_img = tf.keras.preprocessing.image_dataset_from_directory(
    directory="j",
    labels="inferred",
    label_mode="binary",
    color_mode="grayscale",
    batch_size=32,
    shuffle=True,
    seed=the_seed,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
)

print(test_img)

############ model
INPUT_SHAPE = (256, 256, 1)
inp = keras.layers.Input(shape=INPUT_SHAPE)
Conv1 = keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(inp)
Pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(Conv1)
Norm1 = keras.layers.BatchNormalization(axis=-1)(Pool1)
Drop1 = keras.layers.Dropout(rate=0.2)(Norm1)

Conv2 = keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(Drop1)
Pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(Conv2)
Norm2 = keras.layers.BatchNormalization(axis=-1)(Pool2)
Drop2 = keras.layers.Dropout(rate=0.2)(Norm2)

flat = keras.layers.Flatten()(Drop2)

Hidden1 = keras.layers.Dense(512, activation='relu')(flat)
Norm3 = keras.layers.BatchNormalization(axis=-1)(Hidden1)
Drop3 = keras.layers.Dropout(rate=0.2)(Norm3)

Hidden2 = keras.layers.Dense(256, activation='relu')(flat)
Norm4 = keras.layers.BatchNormalization(axis=-1)(Hidden2)
Drop4 = keras.layers.Dropout(rate=0.2)(Norm4)

out = keras.layers.Dense(2, activation='sigmoid')(Drop4)

model = keras.Model(inputs=inp, outputs=out)
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(test_img, epochs=10)

probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])

predictions = probability_model.predict(test_img)
print(predictions)




#pour réaliser des tests avec l'interpreteur qui tourne
while True:
    try:
        command = eval(input("Entrez une commande \n"))
    except:
        pass
