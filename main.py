# Code that runs the main loop of training the TensorFlow models

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

history_drop = model_drop.fit(
    train_images, train_labels, epochs=10, validation_data=(test_images, test_labels)
)

history_l2 = model_drop.fit(
    train_images, train_labels, epochs=10, validation_data=(test_images, test_labels)
)
