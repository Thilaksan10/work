# Introduction to neural network classification with TensorFlow

# Creating data to view and fit
from pandas.core.arrays.categorical import Categorical
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import validation
import tensorflow as tf
from tensorflow.python.ops.gen_logging_ops import histogram_summary
import numpy as np 
from sklearn.metrics import confusion_matrix as cm
import itertools
from tensorflow.keras.datasets import fashion_mnist
import random

from tensorflow.python.tools.module_util import get_parent_dir
from tensorflow.keras.utils import plot_model


# Make 10000 examples
n_samples = 10000
epochs = 1
# Create circles
X, y = make_circles(n_samples, noise=0.03, random_state=42)

# Check out the features and labels
print('X: ', X)
print('y: ', y)

# Visualize data
circles = pd.DataFrame({'X0':X[:,0], 'X1':X[:,1], 'label':y})
print(circles)

# Visualize with a plot
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu)
plt.title('Binary Classification Data')
plt.show()

# Check the shapes of our features and labels
print(X.shape, y.shape)

# View the first example
print(X[0], y[0])
print(X[0].shape)

# Split into training and test data
tf.random.set_seed(42)
X_shuffled = tf.random.shuffle(X, seed=42)
y_shuffled = tf.random.shuffle(y, seed=42)

X_train = X_shuffled[:8000]
y_train = y_shuffled[:8000]

X_test = X_shuffled[8000:]
y_test = y_shuffled[8000:]

# Visualize test and train data
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.RdBu)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.PuOr)
plt.title('Binary Classification Test and Train Data')
plt.show()

# Build a binary classification model
# 1. Create a model
model = tf.keras.Sequential(
    [
        tf.keras.Input(shape=(2,)),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ]
)

# 2. Compile the model
model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.SGD(), metrics=['accuracy'])

# 3. Fit the model
history = model.fit(X_train, y_train, batch_size=64, epochs=epochs)

# Plot history
pd.DataFrame(history.history).plot()
plt.ylabel('epochs')
plt.xlabel('loss')
plt.title('Model Ideal History')
plt.show()

# 4. Evaluate the model
model.evaluate(X_test, y_test)

# Visualize prediction
y_pred = model.predict(X_test)


y_pred_one_hot = tf.round(y_pred)
print(y_pred_one_hot)

def plot_decision_boundary(model, X, y):
    '''
    Plots the decision noundary created by a model predicting on X
    1. https://cs231n.github.io/neural-networks-case-study/
    2. https://github.com/madewithml/basics/blob/master/notebooks/ 
    '''
    # Define the axis boundaries of the plot and create a meshgrid
    x_min, x_max = X[:, 0].numpy().min() - 0.05, X[:, 0].numpy().max() + 0.05
    y_min, y_max = X[:, 1].numpy().min() - 0.05, X[:, 1].numpy().max() + 0.05
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

    # Create X value (we're going to make predictions on these)
    x_in = np.c_[xx.ravel(), yy.ravel()]

    # Make predictions
    y_pred = model.predict(x_in)

    # Check for multi-class
    if len(y_pred[0]) < 1:
        print('doing multiclass classification')
        # We have to reshape our prediction to get them reaady for plotting
        y_pred = np.argmax(y_pred, axis=1).respahe(xx.shape)
    else:
        print('doing binary classification')
        y_pred = np.round(y_pred).reshape(xx.shape)

    # Plot the decision boundary
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    

plt.figure()
plot_decision_boundary(model=model, X=X_test, y=y_pred)
plt.title('Decision Boundary on Test Data')

plt.figure()
plot_decision_boundary(model=model, X=X_test, y=y_pred_one_hot)
plt.title('Decision Boundary on Predictions')

plt.figure()
plot_decision_boundary(model=model, X=X_test, y=y_test)
plt.title('Decision Boundary on Test on-hot encoded Predictions')

plt.show()

for i in range(0, len(y_pred)):
    if int(y_pred_one_hot[i]) != y_test[i]:
        print('Der ' + str(i) + '-te Punkt wird falsch vohergesagt')
        print('Vohersage: ', y_pred[i])
        print('Soll: ', y_test[i])

'''
Finding the best learning rate

To find the ideal learning rate (the learning rate where the loss decreases the most during training) 
we're going to use the folowing steps:
* A learning rate callback - you can think of a callback as an extra piece of functionality, you can add to your
    model while its training
* A modified loss curves plot
'''

# Set random seed
tf.random.set_seed(42)

# Create a model
model_lr = tf.keras.Sequential(
    [
        tf.keras.layers.Input(shape=(2,)),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ]
)

# Compile the model
model_lr.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.SGD(), metrics=['accuracy'])

# Create a learning rate callback
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-3 * 10**(epoch/20))

# Fit the model
history = model_lr.fit(X_train, y_train, batch_size=64, epochs=epochs, callbacks=[lr_scheduler])

pd.DataFrame(history.history).plot()
plt.ylabel('loss')
plt.xlabel('epochs')
plt.title('Model LR History')
plt.show()

# Plot the learning rate versus the loss
lrs = 1e-3 * (10 ** (tf.range(epochs)/20))
plt.figure()
plt.semilogx(lrs, history.history['loss'])
plt.xlabel('Learning Rate')
plt.ylabel('Loss')
plt.title('Learning Rate vs Loss')

# Evaluate the model
model_lr.evaluate(X_test, y_test)
y_pred = model_lr.predict(X_test)
y_pred_one_hot = tf.round(y_pred)

plt.figure()
plot_decision_boundary(model=model_lr, X=X_test, y=y_pred)
plt.title('Decision Boundary on Test Data')

plt.figure()
plot_decision_boundary(model=model_lr, X=X_test, y=y_pred_one_hot)
plt.title('Decision Boundary on Predictions')

plt.figure()
plot_decision_boundary(model=model_lr, X=X_test, y=y_test)
plt.title('Decision Boundary on Test on-hot encoded Predictions')

plt.show()

# Create model with ideal learning rate 0.008
tf.random.set_seed(42)

model_ideal = tf.keras.Sequential(
    [
        tf.keras.layers.Input(shape=(2,)),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ]
)

model_ideal.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.SGD(learning_rate=0.02), metrics=['accuracy'])

history = model_ideal.fit(X_train, y_train, batch_size=64, epochs=epochs)

pd.DataFrame(history.history).plot()
plt.xlabel('epochs')
plt.ylabel('loss')
plt.title('Model Ideal History')
plt.show()

model_ideal.evaluate(X_test, y_test)
y_pred = model_ideal.predict(X_test)
y_pred_one_hot = tf.round(y_pred)

plt.figure()
plot_decision_boundary(model=model_ideal, X=X_test, y=y_test)
plt.title('Decision Boundary on Test Data')

plt.figure()
plot_decision_boundary(model=model_ideal, X=X_test, y=y_pred)
plt.title('Decision Boundary on Predictions')

plt.figure()
plot_decision_boundary(model=model_ideal, X=X_test, y=y_pred_one_hot)
plt.title('Decision Boundary on Test on-hot encoded Predictions')

plt.show()

'''
More classification evaluation methods

Alongside visualizing our models results as much as possible, there are a hadful of other classification
evaluation methods & metrics you should be familiar with:
* Accuracy
* Precision
* Recall
* F1-Score
* Confusion matrix
* Calssification report from (scikit-learn)
'''

# Check accuracy of our model
loss, accuracy = model_ideal.evaluate(X_test, y_test)
print('Model loss on the test set: ', loss)
print('Model accuracy on the test set: ', round(accuracy*100, 2),'%')


def plot_confusion_matrix(y_true, y_pred, classes=None, figsize=(10, 10), text_size=15):
    # Create a confusion matrix
    confusion_matrix = cm(y_true, y_pred)
    print(confusion_matrix)

    confusion_matrix_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
    print(confusion_matrix_normalized)
    n_classes = confusion_matrix.shape[0]

    fig, ax = plt.subplots(figsize=figsize)
    # Create a matrix plot
    cax = ax.matshow(confusion_matrix, cmap=plt.cm.Blues)
    fig.colorbar(cax)

    # Create classes
    if classes:
        labels = classes
    else:
        labels = np.arange(confusion_matrix.shape[0])

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
    for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
        plt.text(j, i, f'{confusion_matrix[i, j]} ({confusion_matrix_normalized[i, j]*100:.2f}%)',
                horizontalalignment='center',
                color='white' if confusion_matrix[i, j] > threshold else 'black',
                size=text_size)
    plt.show()

plot_confusion_matrix(y_test, y_pred_one_hot)

'''
Multiclass Classification

To practice multi-class classification, we're going to build a neural network to classify 
images of different items of cloting.
The data has already been sorted into training and test sets for us
'''
(train_data, train_labels), (test_data, test_labels) = fashion_mnist.load_data()

# Show the first training example
print('Training sample: \n', train_data[0])
print('Training label: \n', train_labels[0])

# Check the shape of a single example
print(train_data[0].shape, train_labels[0].shape)

# Plot a single sample
#plt.imshow(train_data[0])
#plt.show()

# Create a small list so we can index onto our training labels so they're human-readable
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Plot an example image and its label
#plt.imshow(train_data[24])
#plt.title(class_names[train_labels[24]])
#plt.show()

# Plot multiple random images of fashion MNIST
'''
plt.figure()
for i in range(4):
    ax = plt.subplot(2, 2, i+1)
    rand_index = random.choice(range(len(train_data)))
    plt.imshow(train_data[rand_index], cmap=plt.cm.binary)
    plt.title(class_names[train_labels[rand_index]])
    plt.axis(False)
plt.show()
'''

'''
print(train_data.shape)
print(test_data.shape)
X_train = tf.reshape(train_data, shape=(60000, 784))
y_train = tf.one_hot(train_labels, depth=10)
X_test = tf.reshape(test_data, shape=(10000, 784))
y_test = tf.one_hot(test_labels, depth=10)
print(X_train[0][:112])
print(train_data[0])

tf.random.set_seed(42)

model = tf.keras.Sequential(
    [
        tf.keras.layers.Input(shape=(784,)),
        tf.keras.layers.Dense(1000, activation='relu'),
        tf.keras.layers.Dense(1000, activation='relu'),
        tf.keras.layers.Dense(280, activation='relu'),
        tf.keras.layers.Dense(280, activation='relu'),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(28, activation='relu'),
        tf.keras.layers.Dense(28, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ]
)

model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=tf.keras.optimizers.SGD(learning_rate=1e-3), metrics=['accuracy'])

history = model.fit(X_train, y_train, batch_size=32, epochs=50)

pd.DataFrame(history.history).plot()
plt.xlabel('epochs')
plt.ylabel('loss')
plt.title('Multiclass Classification loss curve')
plt.show()

loss, accuracy = model.evaluate(X_test, y_test)
y_pred = model.predict(X_test)

print('Model loss on test set: ', loss)
print('Model accuracy on test set: ', round(accuracy*100, 2), '%')

print(y_test[10])
print(y_pred[10])
'''
'''
batch_size = 32, epochs = 50
loss = 0.5885, accuracy = 88.01%

batch_size = 32, epochs = 100
loss = 0.8654, accuracy = 89.08%
'''
'''
tf.random.set_seed(42)

model_lr = tf.keras.Sequential(
    [
        tf.keras.layers.Input(shape=(784,)),
        tf.keras.layers.Dense(748, activation='relu'),
        tf.keras.layers.Dense(374, activation='relu'),
        tf.keras.layers.Dense(186, activation='relu'),
        tf.keras.layers.Dense(93, activation='relu'),
        tf.keras.layers.Dense(28, activation='relu'),
        tf.keras.layers.Dense(28, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ]
)

model_lr.compile(loss=tf.losses.CategoricalCrossentropy(), optimizer=tf.optimizers.SGD(), metrics=['accuracy'])

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-5 * 10**(epoch/20))

history = model_lr.fit(X_train, y_train, batch_size=64, epochs=100, callbacks=[lr_scheduler])

pd.DataFrame(history.history).plot()
plt.ylabel('loss')
plt.xlabel('epochs')
plt.title('Model LR History')

# Plot the learning rate versus the loss
lrs = 1e-5 * (10 ** (tf.range(100)/20))
plt.figure()
plt.semilogx(lrs, history.history['loss'])
plt.xlabel('Learning Rate')
plt.ylabel('Loss')
plt.title('Learning Rate vs Loss')
plt.show()

'''

# Learning rate should be 0.0011 for Adam loss: 0.5004 accuracy: 84.31%
# Learning rate should be 0.0002 for SGD loss: 0.5561 accuracy: 80.64%

# Set random seed
tf.random.set_seed(42)

model = tf.keras.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=(28,28)),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax'),        
    ]
)

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=tf.keras.optimizers.SGD(learning_rate=0.01), metrics=['accuracy'])

history = model.fit(train_data, train_labels, batch_size=32, epochs=1, validation_data=(test_data, test_labels))

pd.DataFrame(history.history).plot()
plt.xlabel('epochs')
plt.ylabel('loss')
plt.title('Multiclass Classification loss curve')
plt.show()

print(train_data.min(), train_data.max())
print(test_data.min(), test_data.max())

train_data_norm = train_data / train_data.max()
test_data_norm = test_data / test_data.max()

print(train_data_norm.min(), train_data_norm.max())
print(test_data_norm.min(), test_data_norm.max())

tf.random.set_seed(42)

model = tf.keras.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=(28,28)),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax'),
    ]
)

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=tf.keras.optimizers.SGD(learning_rate=0.01), metrics=['accuracy'])

history = model.fit(train_data_norm, train_labels, batch_size=32, epochs=1, validation_data=(test_data_norm, test_labels))

pd.DataFrame(history.history).plot()
plt.xlabel('epochs')
plt.ylabel('loss')
plt.title('Multiclass classification loss curve normalized data')
plt.show()

'''
tf.random.set_seed(42)

model = tf.keras.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=(28,28)),
        tf.keras.layers.Dense(1000, activation='relu'),
        tf.keras.layers.Dense(1000, activation='relu'),
        tf.keras.layers.Dense(280, activation='relu'),
        tf.keras.layers.Dense(280, activation='relu'),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(28, activation='relu'),
        tf.keras.layers.Dense(28, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax'),
    ]
)

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=tf.keras.optimizers.SGD(learning_rate=0.001), metrics=['accuracy'])

history = model.fit(train_data_norm, train_labels, batch_size=32, epochs=50, validation_data=(test_data_norm, test_labels))

pd.DataFrame(history.history).plot()
plt.xlabel('epochs')
plt.ylabel('loss')
plt.title('Multiclass classification loss curve normalized data')
plt.show()

loss, accuracy = model.evaluate(test_data_norm, test_labels)

print('Model loss on test set: ', loss)
print('Model accuracy on test set: ', round(accuracy*100, 2), '%')

# loss: 0.5184 accuracy: 89.76 %
# loss: 0.5921 accuracy: 87.86 %
'''
'''
# Finding the ideal learning rate
tf.random.set_seed(42)

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

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=tf.keras.optimizers.SGD(), metrics=['accuracy'])

# Create the learning rate callback
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-1 * 10**(epoch/20))

lr_history = model.fit(train_data_norm, train_labels, batch_size=32, epochs=25, validation_data=(test_data_norm, test_labels), callbacks=[lr_scheduler])

pd.DataFrame(lr_history.history).plot()
plt.title('Model lr normalized data loss curve')
plt.xlabel('epochs')
plt.ylabel('loss')

plt.figure()
lrs = 1e-1 * (10**(tf.range(25)/20))
plt.semilogx(lrs, lr_history.history['loss'])
plt.xlabel('Learning Rate')
plt.ylabel('loss')
plt.title('Finding ideal learning rate')
plt.show()
'''

# Let's refit the model with the ideal learning rate
tf.random.set_seed(42)

model_ideal = tf.keras.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=(28,28)),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(100, activation='softmax')
    ]
)

model_ideal.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=tf.keras.optimizers.SGD(learning_rate=0.2), metrics=['accuracy'])

history_id = model_ideal.fit(train_data_norm, train_labels, batch_size=32, epochs=20, validation_data=(test_data_norm, test_labels))

pd.DataFrame(history_id.history).plot()
plt.xlabel('epochs')
plt.ylabel('loss')
plt.title('Ideal Model loss curve')
plt.show()

y_probs = model_ideal.predict(test_data_norm)

# Convert all of the prediction probabilitites into integers
y_preds = y_probs.argmax(axis=1)


# Create a confusion matrix
plot_confusion_matrix(y_true=test_labels, y_pred=y_preds, classes=class_names, figsize=(20, 20), text_size=6)

def plot_random_image(model, images, true_labels, classes):
    '''
    Picks a random image, plots it and labels it with prediction and truth label
    '''
    # Set up random integer
    i = random.randint(0, 10000)

    # Create predictions and targets
    target_image = images[i]
    pred_probs = model.predict(target_image.reshape(1, 28, 28))
    print(pred_probs)
    pred_label = classes[pred_probs.argmax()]
    true_label = classes[true_labels[i]]

    # Plot the image
    plt.imshow(target_image, cmap=plt.cm.binary)

    # Change the color of the titles depending on if the prediction is right or wrong
    if pred_label == true_label:
        color = 'green'
    else:
        color = 'red'

    # Add xlabel information (prediction/true label)
    plt.xlabel('Pred: {} {:2.0f}% (True: {})'.format(pred_label, 100*tf.reduce_max(pred_probs), true_label), color=color)
    plt.show()

# Check out a random image as well as its prediction
for i in range(0, 20):
    plot_random_image(model=model_ideal, images=test_data_norm, true_labels=test_labels, classes=class_names)

# Finde th e layers of our most recent model
print(model_ideal.layers)

# Extraact a particular layer
print(model_ideal.layers[1])
# Get the patterns of a layer in our network
weights, biases = model_ideal.layers[1].get_weights()

# Shapes
print('weights: \n', weights)
print('weights shape: ', weights.shape)

# Bias and biases shape
print('biases: \n', biases)
print('biases shape: ', biases.shape)

# Lets view our deep learning model and see the inputs and outputs of each layer
plot_model(model_ideal, show_shapes=True)
plt.show()
