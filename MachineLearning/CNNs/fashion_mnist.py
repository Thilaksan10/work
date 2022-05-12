import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import numpy as np
from sklearn.metrics import confusion_matrix as cm


def plot_confusion_matrix(y_true , y_pred, classes=None, figsize=(10,10), text_size=15):
    # Create confusion matrix
    confusion_matrix = cm(y_true, y_pred)

    confusion_matrix_norm = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
    n_classes = confusion_matrix.shape[0]

    fig, ax = plt.subplots(figsize=figsize)

    # Create matrix plot
    cax = ax.matshow(confusion_matrix, cmap=plt.cm.Blues)
    fig.colorbar(cax)

    # Create classes
    if classes:
        labels = classes
    else:
        labels = np.arange(n_classes)

    # Label the axes
    ax.set(title='Confusion Matrix', 
            xlabel='Predicted Label', 
            ylabel='True Label', 
            xticks=np.arange(n_classes), 
            yticks=np.arange(n_classes), 
            xticklabels=labels, 
            yticklabels=labels)
        
    # Set x-axis labels to bottom
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()

    # Adjust label size
    ax.yaxis.label.set_size(text_size)
    ax.xaxis.label.set_size(text_size)
    ax.title.set_size(text_size)

    # Set threshold for different colors
    threshold = (confusion_matrix.max() + confusion_matrix.min()) / 2

    # Plot the test on each cell
    for i,j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
        plt.text(j, i, 
                f'{confusion_matrix[i, j]} ({confusion_matrix_norm[i, j]*100:.2f}%)',
                horizontalalignment='center',
                color='white' if confusion_matrix[i, j] > threshold else 'black',
                size=text_size
            )
    plt.show()

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
print(X_train.shape)

X_train = X_train / X_train.max()
X_test = X_test / X_test.max()

X_train = X_train.reshape(-1 , 28, 28, 1)
X_test = X_test.reshape(-1 , 28, 28, 1)

model = tf.keras.Sequential(
    [
        tf.keras.layers.Conv2D(32, (3,3), input_shape=(28, 28, 1), activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1000, activation='relu'),
        tf.keras.layers.Dense(1000, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ]
)

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=tf.keras.optimizers.SGD(learning_rate=0.01), metrics=['accuracy'])

history = model.fit(X_train, y_train, batch_size=32, epochs=100, validation_data=(X_test, y_test))

pd.DataFrame(history.history).plot()
plt.xlabel('epochs')
plt.ylabel('loss')
plt.title('Fashion MNIST loss curve')
plt.show()

y_prob = model.predict(X_test)
y_pred = y_prob.argmax(axis=1)

fashion_classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

plot_confusion_matrix(y_test, y_pred, fashion_classes, (13, 13), 5)