import util
import model
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf 
from tensorflow.keras import datasets, layers, models, callbacks
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Code that runs the main loop of training the TensorFlow models

# Specify number of epochs for the two models
epochs_drop = 50
epochs_l2 = 50

# Specify batch size for training
batch_size = 20

# Specify K for K-fold cross validaton
num_folds = 5

# Specify confidence level ( < 1)
cf_lvl = 0.95

train_images, train_labels, test_images, test_labels, img_shape = util.get_images()

stopper = callbacks.EarlyStopping(monitor='val_accuracy', patience=5)

kfold = StratifiedKFold(n_splits=num_folds)

X = np.concatenate((train_images, test_images), axis=0)
y = np.concatenate((train_labels, test_labels), axis=0)

drop_acc = []
drop_loss = []

for train, test in kfold.split(X,y):
    model_drop = model.model_drop_init(img_shape)

    history_drop = model_drop.fit(
        X[train], y[train], epochs=epochs_drop, batch_size=batch_size, callbacks=[stopper], validation_data=(X[test], y[test])
    )

    test_loss, test_acc = model_drop.evaluate(test_images, test_labels, verbose=2)
    drop_loss.append(test_loss*100)
    drop_acc.append(str(test_acc*100) + "%")

l2_acc = []
l2_loss = []

for train, test in kfold.split(X,y):
    model_l2 = model.model_l2_init(img_shape)

    history_l2 = model_l2.fit(
        X[train], y[train], epochs=epochs_drop, batch_size=batch_size, callbacks=[stopper], validation_data=(X[test], y[test])
    )

    test_loss, test_acc = model_drop.evaluate(test_images, test_labels, verbose=2)
    l2_loss.append(test_loss*100)
    l2_acc.append(str(test_acc*100) + "%")

print(drop_acc)
print(drop_loss)

print(l2_acc)
print(l2_loss)

"""
print(
    "\n\nAccuracy: {:.3f} (+/- {:.3f})".format(
        fold_accuracies.mean(), fold_accuracies.std() * 2
    )
)
"""

"""
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
"""
