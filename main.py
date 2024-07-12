import util
import model
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t
import tensorflow as tf 
from tensorflow.keras import datasets, layers, models, callbacks
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import accuracy_score, make_scorer, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Code that runs the main loop of training the TensorFlow models

# Specify number of epochs for the two models
epochs_drop = 100
epochs_l2 = 100

# Specify batch size for training
batch_size = 100

# Specify K for K-fold cross validaton
num_folds = 5

# Specify confidence level ( < 1)
cf_lvl = 0.95
df = num_folds - 1

# Training:

train_images, train_labels, test_images, test_labels, img_shape = util.get_images()

stopper = callbacks.EarlyStopping(monitor='val_loss', patience=5, mode="min")

kfold = StratifiedKFold(n_splits=num_folds)

X_drop = np.concatenate((train_images, test_images), axis=0)
y_drop = np.concatenate((train_labels, test_labels), axis=0)

X_l2 = np.concatenate((train_images, test_images), axis=0)
y_l2 = np.concatenate((train_labels, test_labels), axis=0)

drop_acc = []
drop_loss = []

for train, test in kfold.split(X_drop, y_drop):
    model_drop = model.model_drop_init(img_shape)

    history_drop = model_drop.fit(
        X_drop[train], y_drop[train], epochs=epochs_drop, batch_size=batch_size, 
        callbacks=[stopper], validation_data=(X_drop[test], y_drop[test])
    )

    test_loss, test_acc = model_drop.evaluate(X_drop[test], y_drop[test], verbose=2)
    drop_loss.append(test_loss)
    drop_acc.append(test_acc)

    plt.style.use("ggplot")
    # Plot accuracies vs epoch for model_drop
    plt.plot(history_drop.history["accuracy"], label="accuracy")
    plt.plot(history_drop.history["val_accuracy"], label="val_accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim([0, 1])
    plt.title("model_drop accuracy vs epochs")
    plt.legend(loc="lower right")
    plt.show()

l2_acc = []
l2_loss = []

for train, test in kfold.split(X_l2, y_l2):
    model_l2 = model.model_l2_init(img_shape)

    history_l2 = model_l2.fit(
        X_l2[train], y_l2[train], epochs=epochs_l2, batch_size=batch_size, 
        callbacks=[stopper], validation_data=(X_l2[test], y_l2[test])
    )

    test_loss, test_acc = model_l2.evaluate(X_l2[test], y_l2[test], verbose=2)
    l2_loss.append(test_loss)
    l2_acc.append(test_acc)

    plt.style.use("ggplot")
    # Plot accuracies vs epoch for model_drop
    plt.plot(history_l2.history["accuracy"], label="accuracy")
    plt.plot(history_l2.history["val_accuracy"], label="val_accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim([0, 1])
    plt.title("model_l2 accuracy vs epochs")
    plt.legend(loc="lower right")
    plt.show()
    
print("\n\ndrop model accuracy and loss:")
print(drop_acc)
print(drop_loss)
print("\nl2 model accuracy and loss:")
print(l2_acc)
print(l2_loss)

diff = util.diff_scores(drop_acc, l2_acc)
mean_diff = np.mean(diff)
std_diff = np.std(diff, ddof=df)
cf_interval = t.interval(cf_lvl, df, loc=mean_diff, scale=std_diff/np.sqrt(num_folds))

print(f"\nConfidence Interval = {cf_interval}")
print("\n")

drop_np = np.array(drop_acc)
l2_np = np.array(l2_acc)

print("\n\nDrop Model Accuracy: {:.3f} (+/- {:.3f})".format(drop_np.mean(), drop_np.std() * 2))
print("L2 Model Accuracy: {:.3f} (+/- {:.3f})".format(l2_np.mean(), l2_np.std() * 2))

