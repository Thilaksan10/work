from sklearn import metrics
from sklearn.datasets import make_circles, make_moons
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import validation
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix as cm
from tensorflow.python.ops.gen_array_ops import shape
from tensorflow.python.ops.variables import local_variables
import itertools
from tensorflow.keras.datasets import fashion_mnist
import random
from tensorflow.keras.utils import plot_model

def plot_decision_boundary(model, X, y):
    '''
    Plots the decision noundary created by a model predicting on X
    1. https://cs231n.github.io/neural-networks-case-study/
    2. https://github.com/madewithml/basics/blob/master/notebooks/ 
    '''
    # Define the axis boundaries of the plot and define a meshgrid
    x_min, x_max = X[:, 0].min() - 0, X[:, 0].max() - 0
    y_min, y_max = X[:, 1].min() - 0, X[:, 1].max() - 0
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

    # Create X value
    x_in = np.c_[xx.ravel(), yy.ravel()]

    # Make predictions
    y_pred = model.predict(x_in)

    # Check for type of Classification problem
    if len(y_pred[0]) < 1:
        print('doing multiclass classification')
        y_pred = np.argmax(y_pred, axis=1).reshape(xx.shape)
    else:
        print('doing binary classification')
        y_pred = np.round(y_pred).reshape(xx.shape)

    # Plot the decision boundary
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

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


# 02. Neural network classification with Tensorflow Exercises
'''
1. Play with neural networks in the TensorFlow Playground for 10-minutes. yEspecially try different values of the learning,
    what happens when you decrease it? What happens when you increase it?
'''
print()
print(26 * '-' + ' Excercise 1 ' + 26 * '-')
print()


'''
2. Replicate the model pictured in the TensorFlow Playground diagram below using TensorFlow code.
    Compile it using the Adam optimizer, binary crossentropy loss and accuracy metric. Once it's compiled check
    a summary of the model.
'''
print()
print(26 * '-' + ' Excercise 3 ' + 26 * '-')
print()

# Make 1000 exapmles
X, y = make_circles(1000, noise=0.03, random_state=42)
print(X)
print(y)

# Visualize data
circles = pd.DataFrame({'X0':X[:, 0], 'X1':X[:, 1], 'label':y})
print(circles)

'''
# Visualize with a plot
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdBu)
plt.title('Circle Data')
plt.show()
'''

# Split in 80% training data and 20% testing data
X_train = X[:800]
y_train = y[:800]

X_test = X[800:]
y_test = y[800:]

# Create a model
model = tf.keras.Sequential(
    [
        tf.keras.layers.Input(shape=(2,)),
        tf.keras.layers.Dense(6, activation='relu'),
        tf.keras.layers.Dense(6, activation='relu'),
        tf.keras.layers.Dense(6, activation='relu'),
        tf.keras.layers.Dense(6, activation='relu'),
        tf.keras.layers.Dense(6, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ]
)

model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))

model.summary()


'''
3. Create a classification dataset using Scikit-Learn's make_moons() function, visualize it and then build a model to fit it at 
    over 85% accuracy.
'''
print()
print(26 * '-' + ' Excercise 3 ' + 26 * '-')
print()

X, y = make_moons(10000, shuffle=True, noise=0.03, random_state=42)
print(X)
print(y)

# Viualize Data
moons = pd.DataFrame({'X0':X[:, 0], 'X1':X[:, 1], 'label':y})
print(moons)

'''
## Visualize with a plot
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdBu)
plt.title('Moons data')
plt.show()
'''

# Split data in 80% training data and 20% testing data
X_train = X[:8000]
y_train = y[:8000]

X_test = X[8000:]
y_test = y[8000:]

# Build a model
model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ]
)

model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.SGD(), metrics=['accuracy'])

history = model.fit(X_train, y_train, batch_size=32, epochs=50, validation_data=(X_test, y_test))

# Visualize loss Curve and accuracy curve on training an validation data
pd.DataFrame(history.history).plot()
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()

plt.figure()
plot_decision_boundary(model, X_train, y_train)
plt.figure()
plot_decision_boundary(model, X_test, y_test)
plt.show()

y_pred = model.predict(X_test)

# Plot confusion matrix
plot_confusion_matrix(y_test, tf.round(y_pred))

'''
4. Train a model to get 88%+ accuracy in the MNIST test set. Plot a confusion matrix to see the results after.
'''
print()
print(26 * '-' + ' Excercise 4 ' + 26 * '-')
print()

(X_train, y_train),(X_test, y_test) = fashion_mnist.load_data()

print(X_train[:5])
print(y_train[:5])

print(X_train.shape)
print(y_train.shape)

fashion_classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

X_train_norm = X_train / X_train.max()
X_test_norm = X_test / X_test.max()

model = tf.keras.Sequential(
    [   
        tf.keras.layers.Flatten(input_shape=(28,28)),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ]
)

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=tf.keras.optimizers.SGD(learning_rate=0.1), metrics=['accuracy'])

history = model.fit(X_train_norm, y_train, batch_size=32, epochs=25, validation_data=(X_test_norm, y_test))

pd.DataFrame(history.history).plot()
plt.xlabel('epochs')
plt.ylabel('loss')
plt.title('Fashion MNIST loss curve')
plt.show()

y_prob = model.predict(X_test_norm)
y_pred = y_prob.argmax(axis=1)

plot_confusion_matrix(y_test, y_pred, fashion_classes, (13, 13), 5)

'''
5. Recreate Tensorflow's softmax activation function in your own code. Make sure it can accept a tensor and return that 
    tensor after having the softmax fub´nstion applied to it.
'''
print()
print(26 * '-' + ' Excercise 5 ' + 26 * '-')
print()

def softmax(tensor):
    for x in tensor:
        output = tf.exp(tensor)/ tf.reduce_sum(tf.exp(tensor))
        return output

tensor = tf.random.normal(shape=(32, 10))
outputs = tf.keras.activations.softmax(tensor)
print(outputs)
sum_of_outputs = tf.reduce_sum(outputs[0, :])
print(sum_of_outputs)


'''
6. Create a function (or write code) to visualize multiple image predictions for the MNIST at the same time. Plot at least three different 
    images and their prediction labels at the same time.
    Hint: see the classification tutorial in the TensorFlow documentation for ideas.
'''
print()
print(26 * '-' + ' Excercise 6 ' + 26 * '-')
print()

def plot_random_images(model, images, true_labels, classes):
    # Set up random integer
    fig = plt.figure(figsize=(10, 10))
    ax = fig.subplots(1, 3, sharex=False)
    for i in range(0, 3):
        rand = random.randint(0, len(images))

        # Create prediction and target
        target_image = images[rand]
        pred_prob = model.predict(target_image.reshape(1, 28, 28))
        pred_label = classes[pred_prob.argmax()]
        true_label  = classes[true_labels[rand]]

        # Plot the image
        ax[i].imshow(target_image, cmap=plt.cm.binary)

        # Change the color of the title depending on if the prediction is wrong or right
        if true_label == pred_label:
            color = 'green'
        else:
            color = 'red'

        # Add xlabel Information
        ax[i].set_title('Pred: {} {:.2f}% (True: {})'.format(pred_label, 100*tf.reduce_max(pred_prob), true_label), color=color)
        ax[i].title.set_size(8)

for i in range(0,20):
    plot_random_images(model, X_test_norm, y_test, fashion_classes)
    plt.show()

'''
7. Make function to show an image of a certain class of the fashion MNIST dataset and make a prediction on it.
    For example, plot 3 images of the T-shirt class with their predictions.
'''
print()
print(26 * '-' + ' Excercise 7 ' + 26 * '-')
print()

def plot_and_predict_class(model, images, true_labels, classes, fashion_class):
    fashion_class_images = []
    fashion_class_labels = []
    for i in range(0, len(images)):
        if fashion_class == true_labels[i]:
            fashion_class_images.append(images[i])
            fashion_class_labels.append(fashion_class)
    
    plot_random_images(model, fashion_class_images, fashion_class_labels, classes)

for i in range(0,20):
    plot_and_predict_class(model, X_test_norm, y_test, fashion_classes, i%10)
    plt.show()
