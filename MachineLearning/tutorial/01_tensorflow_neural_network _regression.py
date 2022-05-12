# Introduction to Regression with Neural Networks in Tensorflow
# Import TensorFlow
import tensorflow as tf
print(tf.__version__)

# Creating data to view and fit
import numpy as np
import matplotlib.pyplot as plt

# Create features
X = np.array([-7.0, -4.0, -1.0, 2.0, 5.0, 8.0, 11.0, 14.0])

# Create labels 
y = np.array([3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0])

# Visualize it
plt.scatter(X, y)
#plt.show()

# Input and Output shapes
# Create a demo tensor for our housing price prediction problem
house_info = tf.constant(['bedroom', 'bathroom', 'garage'])
house_price = tf.constant([939700])
print('Hosue Info: ', house_info)
print('Hosue Price: ', house_price)

input_shape = X.shape
output_shape = y.shape
print('Input Shape: ', input_shape)
print('Output Shape: ', output_shape)

# Turn our NumPy arrays into tensors
X = tf.cast(tf.constant(X), dtype=tf.float32)
y = tf.cast(tf.constant(y), dtype=tf.float32)
print('X: ', X)
print('y: ', y)
input_shape = X[0].shape
output_shape = y[0].shape
print('Input Shape: ', input_shape)
print('Output Shape: ', output_shape)

'''
Steps in modelling with Tensorflow
1. Creating a model - define the input and output layers, as well as the hidden layers of a deep learning model.
2. Compiling a model - define the loss function (tells our model how wrong it is) and the optimizer (tells our model how to improve 
    the patterns its learning) and evaluation metrics (what we can use to interpret the performance of our model).
3. Fitting a model - letting the model try to find patterns between X & y (features an labels)
'''

# Set random seed
tf.random.set_seed(42)

# 1. Create a model using the Sequential API
model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(1)
    ]
)

# 2. Compile the model
# mae = mean absolute error
# sgd = stochastic gradient descent
model.compile(loss=tf.keras.losses.mae, optimizer=tf.keras.optimizers.SGD(), metrics=['mae'])

# 3. Fit the model
model.fit(X, y, epochs=5)

# Check out X and y
print('X: ', X)
print('y: ', y)

# Try and make a prediction using our model
print('Prediction: ', model.predict([17.0]))

'''
Improving our model
We can improve our model, by altering the steps we took to create a model
1. Create a model - here we might add more layers, increase the number of hidden units (neurons) within each of the hidden layers, 
    change the activation function of each layer.
2. Compiling a model - here we might change the optmization function or perhaps the learning rate of the optimization function.
3. Fitting a model - here we might fit a modle for more epochs (leave it training for longer) or on more data 
    (give the model more examples to learn from)
'''

# Lets rebuild our model
# 1. Create a model (secified to your problem)
model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(1)
    ]
)

# 2. Compile the model
model.compile(loss=tf.keras.losses.mae, optimizer=tf.keras.optimizers.SGD(), metrics=['mae'])

# 3. Fit the model (this time we'll train longer)
model.fit(X, y, epochs=100)

# Check out X and y
print('X: ', X)
print('y: ', y)

# Let's see if our model's prediction has improved
print('Prediction: ', model.predict([17.0]))

# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# Let's see if we can make further changes to improve our model
# 1. Create the model (this time with an extra hidden layer with 100 units)
model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(1)
    ]
)

# 2. Compile the model
model.compile(loss=tf.keras.losses.mae, optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), metrics=['mae'])

# 3. Fit the model
model.fit(X, y, epochs=100)

# Check out X and y
print('X: ', X)
print('y: ', y)

# Let's see if our model's prediction has improved
print('Prediction: ', model.predict([17.0]))

# Evaluating a model
# Make a bigger dataset
X = tf.range(-100, 100, 4)
print('X: ', X)

# Make labels for the dataset
y = X + 10
print('y: ', y)

# Visualize the data
plt.scatter(X,y)
#plt.show()

''' 
The 3 Sets ...
* Training -set - the model learns from this data, which is typically 70-80% of the total data you have available.
* Validation set - the model gets tuned on this data, which is typically 10-15% of the data available.
* Test set - the model gets evaluated on this data to test what it has learned, this set is typically 10-15% of th total data avaialble.
'''

# Chack the langth of how many samples we have
print('Data: ', len(X))

# Split the data into train and test sets
# first 40 are training samples (80% of the data)
X_train = X[:40]
X_test = X[40:]
y_train = y[:40]
y_test = y[40:]

print('X_train: ', X_train)
print('y_train: ', y_train)

print('X_test: ', X_test)
print('X_test: ', X_test)

# Visualizing the data
plt.figure(figsize=(10,7))
# PLot training data in blue
plt.scatter(X_train, y_train, c='b', label='Training data')
plt.scatter(X_test, y_test, c='g', label='Testing data')
# Show a legend
plt.legend()
plt.show()

# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# Let's have a look at how to build a neural network for our data
# 1. Create a model
model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(1)
    ]
)

# 2. Compile the model
model.compile(loss=tf.keras.losses.mae, optimizer=tf.keras.optimizers.SGD(), metrics=['mae'])

# 3. Fit the model
#model.fit(X_train, y_train, epochs=100)

# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# Let's create a model which builds automaticaly by definin the input_shape argument
tf.random.set_seed(42)
# 1. Create a model (smae as above)
model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(10, input_shape=[1], name='input_layer'),
        tf.keras.layers.Dense(1, name='output_layer')
    ],
    name='model_1'
)

# 2. Compile the model (same as above)
model.compile(loss=tf.keras.losses.mae, optimizer=tf.keras.optimizers.SGD(), metrics=['mae'])

model.summary()

# 3. Fit model to training data
model.fit(X_train, y_train, epochs=100, verbose=0)

# Get Summary of out model
model.summary()

from tensorflow.keras.utils import plot_model
plot_model(model=model, show_shapes=True)

# Visualizing our model's prediction
# Make some predictions
y_pred = model.predict(X_test)
print('y_pred: ', y_pred)
print('y_test: ', y_test)

# Let's create a plotting function
def plot_predictions(train_data=X_train, train_labels=y_train, test_data=X_test, test_labels=y_test, predictions=y_pred):
    '''
    Plots training data, test data and compares predictions to ground truth labels.
    '''
    plt.figure(figsize=(10,7))
    # PLot training data in blue
    plt.scatter(train_data, train_labels, c='b', label='Training data')
    # Plot testing data on green
    plt.scatter(test_data, test_labels, c='g', label='Testing data')
    # Plot model's predictions in red
    plt.scatter(test_data, predictions, c='r', label='Predictions')
    # Show the Legend
    plt.legend()
    plt.show()

plot_predictions(train_data=X_train, train_labels=y_train, test_data=X_test, test_labels=y_test, predictions=y_pred)

# Evaluating our model's predictions with regression evaluation metrics
# Evaluate the model on the test set
model.evaluate(X_test, y_test)

# Calculate the mean absolute error
mae = tf.metrics.mean_absolute_error(y_true=y_test, y_pred=tf.constant(y_pred))
print('MAE: ', mae)

# Calculate the mean absolute error
mae = tf.metrics.mean_absolute_error(y_true=y_test, y_pred=tf.squeeze(y_pred))
print('MAE: ', mae)

# Calculate the mean square error
mse = tf.metrics.mean_squared_error(y_true=y_test, y_pred=tf.squeeze(y_pred))
print('MSE: ', mse)

# Calculate the huber
huber = tf.keras.losses.huber(y_true=y_test, y_pred=tf.squeeze(y_pred))
print('HUBER: ', huber)

# Make some functions to reuse MAE and MSE
def mae(y_true, y_pred):
    print('Calculate MAE')
    return tf.metrics.mean_absolute_error(y_true=y_true, y_pred=y_pred)

def mse(y_true, y_pred):
    return tf.metrics.mean_squared_error(y_true=y_true, y_pred=y_pred)

# Runnning experiments to improve our model
'''
1. Get more data - get more examples for your model to train on ( more opportunities to learn paterns or relationships between features and labels).
2. Make your model larger (using a more complex model) - this might come in the form of more layers or more hidden units in each layer.
3. Train for longer - give your model more of a chance to find patterns in the data. 

Let's do 3 modelling experiments:

1. 'model_1' - same as the original model, 1 layer, trained for 100 epochs.
2. 'model_2' - 2 layers, trained for 100 epochs
3. 'model_3' - 2 layers, from which the first hidden layer uses relu as activation function, trained for 500 epochs
'''

# Build model_1
# Set random seed
tf.random.set_seed(42)
# 1. Create the model
model_1 = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(1)
    ]
)

# 2. Compile the model
model_1.compile(loss=tf.keras.losses.mae, optimizer=tf.keras.optimizers.SGD(), metrics=['mae'])

# 3. Fit the model
model_1.fit(X_train, y_train, epochs=100, verbose=0)

# Make and plot predictions fpr model_1
y_pred_1 = model_1.predict(X_test)
plot_predictions(predictions=y_pred_1)

# Calculate model_1 evalutaion metrics
mae_1 = mae(y_test, tf.squeeze(y_pred_1))
mse_1 = mse(y_test, tf.squeeze(y_pred_1))

print('Mae_1: ', mae_1)
print('Mse_1: ', mse_1)

# ------------------------------------------
# ------------------------------------------
# ------------------------------------------
# ------------------------------------------

# Build model_2
# Set random seed
tf.random.set_seed(42)
# 1. Create a model
model_2 = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(10),
        tf.keras.layers.Dense(1)
    ]
)

# 2. Compile a model
model_2.compile(loss=tf.keras.losses.mae, optimizer=tf.keras.optimizers.SGD(), metrics=['mae'])

# 3. Fit the model
model_2.fit(X_train, y_train, epochs=100, verbose=0)

# Make and plot predictions for model_2
y_pred_2 = model_2.predict(X_test)
plot_predictions(predictions=y_pred_2)

# Calculate model_2 evaluation metrics
mae_2 = mae(y_test, tf.squeeze(y_pred_2))
mse_2 = mse(y_test, tf.squeeze(y_pred_2))

print('Mae_2: ', mae_2)
print('Mse_2: ', mse_2)

# ------------------------------------------
# ------------------------------------------
# ------------------------------------------
# ------------------------------------------

# Build model_3
# Set random seed
tf.random.set_seed(42)
# 1. Create a model
model_3 = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(1)
    ]
)

# 2. Compile the model
model_3.compile(loss=tf.keras.losses.mae, optimizer=tf.keras.optimizers.SGD(), metrics=['mae'])

# 3. Fit the model
model_3.fit(X_train, y_train, epochs=500, verbose=0)

# Make and plot prediction for model_3
y_pred_3 = model_3.predict(X_test)
plot_predictions(predictions=y_pred_3)

# Calculate model_3 evaluation metrics
mae_3 = mae(y_test, tf.squeeze(y_pred_3))
mse_3 = mse(y_test, tf.squeeze(y_pred_3))

print('Mae_3: ', mae_3)
print('Mse_3: ', mse_3)

# Comparing te results of our experimenrs
#Let's compare our model's results using a pandas DataFrame
import pandas as pd

model_results = [
    ['model_1', mae_1.numpy(), mse_1.numpy()],
    ['model_2', mae_2.numpy(), mse_2.numpy()],
    ['model_3', mae_3.numpy(), mse_3.numpy()]
]

all_results = pd.DataFrame(model_results, columns=['model', 'mae', 'mse'])
print(all_results)

model_2.summary()

# Saving our models
# Saving our models allows us to use them outside of where they were trained, such as in a web application or mobile app.

# Save model using the SavedModel format
model_3.save('best_model_SavedModel_format')

# Save model using the HDF5 format
model_3.save('best_model_HDF5_format.h5')

# Loading in a saved model
# load in the SavedModel format model
loaded_SavedModel_format = tf.keras.models.load_model('best_model_SavedModel_format')
loaded_SavedModel_format.summary()

# Compare model_3 predictions with SavedModel format predictions, mae and mse
model_3_pred = model_3.predict(X_test)
loaded_SavedModel_format_pred = loaded_SavedModel_format.predict(X_test)
print(16 * '-' + 'Compare loaded SavedModel with model_3' + 16 * '-')
print('Are both predictions equal? \n ', model_3_pred == loaded_SavedModel_format_pred)
print('Are both mae equal? \n ', mae(y_true=y_test, y_pred=tf.squeeze(model_3_pred)) == mae(y_true=y_test, y_pred=tf.squeeze(loaded_SavedModel_format_pred)))
print('Are both mse equal? \n ', mse(y_true=y_test, y_pred=tf.squeeze(model_3_pred)) == mse(y_true=y_test, y_pred=tf.squeeze(loaded_SavedModel_format_pred)))

# Load in a model using the .h5 format
loaded_h5_model = tf.keras.models.load_model('best_model_HDF5_format.h5')
loaded_h5_model.summary()

# Compare model_3 predictions with HDF5 format predictions, mae and mse
loaded_h5_model_pred = loaded_h5_model.predict(X_test)
print(16 * '-' + 'Compare loaded HDF5 with model_3' + 16 * '-')
print('Are both predictions equal? \n ', model_3_pred == loaded_h5_model_pred)
print('Are both mae equal? \n ', mae(y_true=y_test, y_pred=tf.squeeze(model_3_pred)) == mae(y_true=y_test, y_pred=tf.squeeze(loaded_h5_model_pred)))
print('Are both mse equal? \n ', mse(y_true=y_test, y_pred=tf.squeeze(model_3_pred)) == mse(y_true=y_test, y_pred=tf.squeeze(loaded_h5_model_pred)))

# Excercise
print()
print(20 * '-' + 'Excercise' + 20 * '-')
print()

# 1. Create your own regression dataset (or make the one we created in 'Create data to view and fit' bigger) and build fit a model to it.

# Set random seed
tf.random.set_seed(24)

# Create data and label
X = tf.random.shuffle(tf.range(-10000, 10000, 4), seed=24)
y = (X**3) + 5 
print('X: ', X)
print('y: ', y)

# Visualize it
plt.scatter(X, y)
#plt.show()

# Split data into training and test set
X_train = X[:4000]
y_train = y[:4000]
X_test = X[4000:]
y_test = y[4000:]

print('X_train: ', X_train)
print('X_test: ', X_test)

# Build neural network
# 1. Create a model
model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(1, activation='relu')
    ]
)

# 2. Compile the model
model.compile(loss=tf.keras.losses.mae, optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['mae'])

# 3. Fit the model
history = model.fit(X_train, y_train, batch_size=32, epochs=100)

# Plot history (also known as a loss curve or a training curve)
pd.DataFrame(history.history).plot()
plt.ylabel('loss')
plt.xlabel('epochs')
plt.show()

# Make and plot prediction for model_3
y_pred = model.predict(X_test)
plot_predictions(train_data=X_train, train_labels=y_train, test_data=X_test, test_labels=y_test, predictions=y_pred)

# Calculate evaluation metrics
mae = mae(y_test, tf.squeeze(y_pred))
mse = mse(y_test, tf.squeeze(y_pred))

print('Mae: ', mae)
print('Mse: ', mse)

# Try building a neural network with 4 Dense layers and fitting it to your own regression dataset, how does it perform?
# 1. Create a model
modified_model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(1, activation='relu')
    ]
)

# 2. Compile the model
modified_model.compile(loss=tf.keras.losses.mae, optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['mae'])

# 3. Fit the model
history = modified_model.fit(X_train, y_train, batch_size=64, epochs=500, verbose=1)

# Plot history (also known as a loss curve or a training curve)
pd.DataFrame(history.history).plot()
plt.ylabel('loss')
plt.xlabel('epochs')
plt.show()

# Make and plot prediction for modified_model
y_pred = modified_model.predict(X_test)
plot_predictions(train_data=X_train, train_labels=y_train, test_data=X_test, test_labels=y_test, predictions=y_pred)

# Calculate evaluation metrics
mae_3 = tf.metrics.mean_absolute_error(y_true=y_test, y_pred=tf.squeeze(y_pred))
mse_3 = tf.metrics.mean_squared_error(y_true=y_test, y_pred=tf.squeeze(y_pred))
#196426.84, 502842.25, 234143.3
print('Mae: ', mae_3)
print('Mse: ', mse_3)

# 4. Import the Boston pricing dataset from Tensorflow tf.keras.datasets and model it.
print()
print(16 * '-' + ' Boston Pricing Dataset ' + 16 * '-') 
print()
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.boston_housing.load_data(
    path='boston_housing.npz', test_split=0.2, seed=42
)

print('X_train: ', X_train, len(X_train))
print('X_test: ', X_test, len(X_test))

print('y_train: ', y_train, len(y_train))
print('y_test: ', y_test, len(y_test))

# 1. Create a model
model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(1, activation='relu')
    ]
)

# 2. Compile the model
model.compile(loss=tf.keras.losses.mae, optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001), metrics=['mae'])

# 3. Fit the model
history = model.fit(X_train, y_train, batch_size=8, epochs=10000)

# Plot history (also known as a loss curve or a training curve)
pd.DataFrame(history.history).plot()
plt.ylabel('loss')
plt.xlabel('epochs')
plt.show()


# Make and plot prediction for modified_model
y_pred = model.predict(X_test)
#plot_predictions(train_data=X_train, train_labels=y_train, test_data=X_test, test_labels=y_test, predictions=y_pred)
plt.figure(figsize=(10,7))
data = list(range(0,102))
print(data, len(data))
print(y_test, len(y_test))
print(y_pred, len(y_pred))
plt.scatter(data, y_test, c='g', label='True')
plt.scatter(data, y_pred, c='r', label='Prediction')
# Show the Legend
plt.legend()
plt.show()
#3.702801, 3.0768383, 2.5399284, 2.4654446, 2.46536, 2.3199942, 2.263697, 2.268006
# Calculate evaluation metrics
mae_3 = tf.metrics.mean_absolute_error(y_true=y_test, y_pred=tf.squeeze(y_pred))
mse_3 = tf.metrics.mean_squared_error(y_true=y_test, y_pred=tf.squeeze(y_pred))

print('Mae: ', mae_3)
print('Mse: ', mse_3)


