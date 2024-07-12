import util
import tensorflow as tf 
from tensorflow.keras import layers, models, regularizers

#TensorFlow code that defines the network

# Create cnn model with 3 layers of convolution and 2 for pooling. Uses dropout regularizer
def model_drop_init(image_shape):
    model_drop = models.Sequential() # Instantiate sequential with the name model
    # Add convolutional layers to Model_drop
    model_drop.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=image_shape)) # 2D convolutional layer with 32 filters, (3x3 convolutions), (diff from image sizes)
    model_drop.add(layers.MaxPooling2D((2, 2))) # Drags a 2x2 kernal across its input, chooses the max for each one, go to output
    model_drop.add(layers.Dropout(0.2))
    model_drop.add(layers.Conv2D(64, (3, 3), activation="relu")) # Convolutional layer also 3x3, 62 filters, 
    model_drop.add(layers.MaxPooling2D((2, 2))) # 2x2 pooling
    model_drop.add(layers.Dropout(0.2))
    model_drop.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model_drop.add(layers.Flatten()) # Flattens 2 dimesnonal input into a 1 dimensional
    model_drop.add(layers.Dropout(0.2))
    model_drop.add(layers.Dense(64, activation="relu")) # Regular dense layer of 64 nodes, relu activation
    model_drop.add(layers.Dropout(0.2))
    model_drop.add(layers.Dense(64, activation="relu"))
    model_drop.add(layers.Dense(100, activation="softmax")) # Output dense layer where # of nodes = # of classes 

    model_drop.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )
    return model_drop

# Create cnn model with 2 layers of convolution and pooling and l2 regularizer
def model_l2_init(image_shape):
    model_l2 = models.Sequential() # Instantiate sequential with the name model
    # Add convolutional layers to Model_l2
    model_l2.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=image_shape, kernel_regularizer=regularizers.l2(0.0001))) # 2D convolutional layer with 32 filters, (3x3 convolutions), (diff from image sizes)
    model_l2.add(layers.MaxPooling2D((2, 2))) # Drags a 2x2 kernal across its input, chooses the max for each one, go to output
    model_l2.add(layers.Conv2D(64, (3, 3), activation="relu", kernel_regularizer=regularizers.l2(0.0001))) # Convolutional layer also 3x3, 62 filters, 
    model_l2.add(layers.MaxPooling2D((2, 2))) # 2x2 pooling
    model_l2.add(layers.Conv2D(64, (3, 3), activation="relu", kernel_regularizer=regularizers.l2(0.0001))) 
    model_l2.add(layers.Flatten()) # Flattens 2 dimesnonal input into a 1 dimensional
    model_l2.add(layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(0.0001))) # Regular dense layer of 64 nodes, relu activation
    model_l2.add(layers.Dense(100, activation="softmax")) # Output dense layer where # of nodes = # of classes 

    model_l2.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )
    return model_l2
