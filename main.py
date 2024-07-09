import util
import model
import tensorflow as tf 
from tensorflow.keras import datasets, layers, models

# Code that runs the main loop of training the TensorFlow models

# Instantiate sequential with the name model

# Specify number of epochs for the two models
epochs_drop = 50
epochs_l2 = 50

train_images, train_labels, test_images, test_labels, image_shape = util.get_images()

model_drop = model.model_drop_init(image_shape)
model_l2 = model.model_l2_init(image_shape)

# Output summaries of each of the models
model_drop.summary()
model_l2.summary()

# Compile both models

model_drop.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

model_l2.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

stopper = callbacks.EarlyStopping(monitor='val_accuracy', patience=5)

history_drop = model_drop.fit(
    train_images, train_labels, epochs=epochs_drop, callbacks=[stopper], validation_data=(test_images, test_labels)
)

history_l2 = model_drop.fit(
    train_images, train_labels, epochs=epochs_l2, callbacks=[stopper], validation_data=(test_images, test_labels)
)

