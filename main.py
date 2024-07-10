import util
import model
import matplotlib.pyplot as plt
import tensorflow as tf 
from tensorflow.keras import datasets, layers, models, callbacks

# Code that runs the main loop of training the TensorFlow models

# Instantiate sequential with the name model

# Specify number of epochs for the two models
epochs_drop = 20
epochs_l2 = 20

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
    train_images, train_labels, epochs=epochs_drop, validation_data=(test_images, test_labels)
)

history_l2 = model_l2.fit(
    train_images, train_labels, epochs=epochs_l2, validation_data=(test_images, test_labels)
)

plt.style.use("ggplot")

# Plot accuracies vs epoch for model_drop
plt.plot(history_drop.history["accuracy"], label="accuracy")
plt.plot(history_drop.history["val_accuracy"], label="val_accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.ylim([0, 1])
plt.title("model_drop accuracy vs epochs")
plt.legend(loc="lower right")
test_loss, test_acc = model_drop.evaluate(test_images, test_labels, verbose=2)

# Plot accuracies vs epoch for model_l2
plt.plot(history_l2.history["accuracy"], label="accuracy")
plt.plot(history_l2.history["val_accuracy"], label="val_accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.ylim([0, 1])
plt.title("model_l2 accuracy vs epochs")
plt.legend(loc="lower right")
test_loss, test_acc = model_l2.evaluate(test_images, test_labels, verbose=2)

