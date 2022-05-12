# A Larger Example
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# Read in the insurance dataset
insurance = pd.read_csv('https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv')
print(insurance)

# Let's try one-hot encode our DataFrame so it's all numbers
insurance_one_hot = pd.get_dummies(insurance)
print(insurance_one_hot.head())

# Create X & y values (features and labels)
X = insurance_one_hot.drop('charges', axis=1)
y = insurance_one_hot['charges']
print('X: \n', X)
print('y: \n', y)

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print('Lenght of X: ', len(X))
print('Lenght of X_train: ', len(X_train))
print('Lenght of X_test: ', len(X_test))

# Build a neural network
tf.random.set_seed(42)
# 1. Create the model
model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(1)
    ]
)

# 2. Compile the model
model.compile(loss=tf.keras.losses.mae, optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), metrics=['mse'])

# 3. Fit the model
model.fit(X_train, y_train, epochs=500)

# Check the results of insurance model on the test data
model.evaluate(X_test, y_test)

'''
To try to improve our model, we'll reun 2 experiments:
1. Add an extra layer with more hidden units
2. Train for longer
'''
# Set random seed
tf.random.set_seed(42)
# 1. Create the model
model_2 = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(1)
    ]
)

# 2. Compile the model
model_2.compile(loss=tf.keras.losses.mae, optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), metrics=['mse'])

# 3. Fit the model
model_2.fit(X_train, y_train, epochs=500, verbose=0)

# Evaluate larger model
model_2.fit(X_test, y_test)

# Set random seed
tf.random.set_seed(42)
# 1. Create the model
model_3 = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(1)
    ]
)

# 2. Compile the model
model_3.compile(loss=tf.keras.losses.mae, optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), metrics=['mae'])

# 3. Fit the model
history = model_3.fit(X_train, y_train, epochs=1500)

# 4. Evaluate the model which trained longer
model_3.evaluate(X_test, y_test)

# Plot history (also known as a loss curve or a training curve)
pd.DataFrame(history.history).plot()
plt.ylabel('loss')
plt.xlabel('epochs')
plt.show()

# Preprocessing data (normalization and standardization)
'''
In terms of scaling values, neural networks tend to prefer normalization. 

If you're not sure on which to use, you could try both and see which performs better.
'''

X['age'].plot(kind='hist')
plt.show()
X['bmi'].plot(kind='hist')
plt.show()
print(X['children'].value_counts())

# Create a column tranformer
# Turn all values in colums age, bmi and children between 0 and 1.
ct = make_column_transformer(
    (MinMaxScaler(), ['age', 'bmi', 'children']),
    (OneHotEncoder(handle_unknown='ignore'), ['sex', 'smoker', 'region'])
)

# Create X and y
X = insurance.drop('charges', axis=1)
y = insurance['charges']

# Build our train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the column transformer to our training data
ct.fit(X_train)

# Transform training and test data with normalization (MinMaxScaler) and OneHotEncoder
X_train_normal = ct.transform(X_train)
X_test_normal = ct.transform(X_test)

# What does our data look like now?
print('X_train: ', X_train.loc[0])
print('X_train_normal: ', X_train_normal[0])

# Build a neural network model on normalized and one-hot encoded data
# Set random seed
tf.random.set_seed(42)

# 1. Create a model
model_4 = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(1)
    ]
)

# 2. Compile a model
model_4.compile(loss=tf.keras.losses.mae, optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), metrics=['mae'])

# 3.Fit the model
history_normal = model_4.fit(X_train_normal, y_train, epochs=500)

# 4.. Evaluate the model
model_4.evaluate(X_test_normal, y_test)

# Plot history (also known as a loss curve or a training curve)
pd.DataFrame(history_normal.history).plot()
plt.ylabel('loss')
plt.xlabel('epochs')
plt.show()
