#TensorFlow code that defines the network

# Instantiate sequential with the name model
model_drop = models.Sequential()
model_l2 = models.Sequential()

# Add convolutional layers to Model_drop
model_drop.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=image_shape)) # 2D convolutional layer with 32 filters, (3x3 convolutions), (diff from image sizes)
model_drop.add(layers.MaxPooling2D((2, 2))) # Drags a 2x2 kernal across its input, chooses the max for each one, go to output
model_drop.add(layers.Conv2D(64, (3, 3), activation="relu")) # Convolutional layer also 3x3, 62 filters, 
model_drop.add(layers.MaxPooling2D((2, 2))) # 2x2 pooling
model_drop.add(layers.Conv2D(64, (3, 3), activation="relu"))
model_drop.add(layers.Flatten()) # Flattens 2 dimesnonal input into a 1 dimensional
model_drop.add(layers.Dense(64, activation="relu")) # Regular dense layer of 64 nodes, relu activation
model_drop.add(layers.Dense(100)) # Output dense layer where # of nodes = # of classes 

# Add convolutional layers to Model_l2
model_l2.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=image_shape)) # 2D convolutional layer with 32 filters, (3x3 convolutions), (diff from image sizes)
model_l2.add(layers.MaxPooling2D((2, 2))) # Drags a 2x2 kernal across its input, chooses the max for each one, go to output
model_l2.add(layers.Conv2D(64, (3, 3), activation="relu")) # Convolutional layer also 3x3, 62 filters, 
model_l2.add(layers.MaxPooling2D((2, 2))) # 2x2 pooling
model_l2.add(layers.Conv2D(64, (3, 3), activation="relu")) 
model_l2.add(layers.Flatten()) # Flattens 2 dimesnonal input into a 1 dimensional
model_l2.add(layers.Dense(64, activation="relu")) # Regular dense layer of 64 nodes, relu activation
model_l2.add(layers.Dense(100)) # Output dense layer where # of nodes = # of classes 

