import numpy as np
from matplotlib import pyplot as plt
from tensorflow import keras

from utility import load, process_image
from imutils import paths
import tensorflow as tf

img_path = "./Image_Dataset/"

image_paths = list(paths.list_images(img_path))

# Load the image data and labels
train_images, train_labels, test_images, test_labels = load(image_paths, verbose=100)

CLASS_NAMES = ['Prehypertension', 'Stage-0 Hypertension', 'Stage-2 Hypertension', 'Stage-3 Hypertension']

train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))

train_ds_size = tf.data.experimental.cardinality(train_ds).numpy()
test_ds_size = tf.data.experimental.cardinality(test_ds).numpy()
print('Train size:', train_ds_size)
print('Test size:', test_ds_size)

train_ds = (train_ds
            .map(process_image)
            .shuffle(buffer_size=train_ds_size)
            .batch(batch_size=32, drop_remainder=True)
            )
test_ds = (test_ds
           .map(process_image)
           .shuffle(buffer_size=test_ds_size)
           .batch(batch_size=32, drop_remainder=True)
           )

model = keras.models.Sequential([
    keras.layers.Conv2D(filters=128, kernel_size=(11, 11), strides=(4, 4), activation='relu',
                        input_shape=(64, 64, 3)),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(2, 2)),
    keras.layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3, 3)),
    keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(1024, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1024, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation='softmax')

])

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.optimizers.SGD(learning_rate=0.001),
    metrics=['accuracy']
)
print(model.summary())

history = model.fit(
    train_ds,
    epochs=150,
    validation_data=test_ds,
    validation_freq=1
)
model.history.history.keys()

f, ax = plt.subplots(2, 1, figsize=(10, 10))

# Assigning the first subplot to graph training loss and validation loss
ax[0].plot(model.history.history['loss'], color='b', label='Training Loss')
ax[0].plot(model.history.history['val_loss'], color='r', label='Validation Loss')

# Plotting the training accuracy and validation accuracy
ax[1].plot(model.history.history['accuracy'], color='b', label='Training  Accuracy')
ax[1].plot(model.history.history['val_accuracy'], color='r', label='Validation Accuracy')

plt.legend()
print('Accuracy Score = ', np.max(history.history['val_accuracy']))
