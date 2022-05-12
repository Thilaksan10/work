# Introduction to Tensors
# Import Tensorflow
import tensorflow as tf
from tensorflow._api.v2 import random
import tensorflow_probability as tfp

print(tf.__version__)

# Create tensors with tf.constant()
scalar = tf.constant(7)
print('Scalar: ', scalar)

# Check the number of dimensions of a tensor (ndim stands for number of dimensions)
print('Scalar dimensions: ', scalar.ndim)

# Create a vector
vector = tf.constant([10, 10])
print('Vector: ', vector)

# Check the dimensions of our vector
print('Vector dimesions:', vector.ndim)

# Create a matrix (has more than on dimensions)
matrix = tf.constant(
    [
        [10, 7],
        [7, 10]
    ]
)
print('Matrix: ', matrix)

print('Matrix dimensions: ', matrix.ndim)

# Create another matrix
another_matrix = tf.constant(
    [
        [10., 7.],
        [3., 2.],
        [8., 9.]
    ],
    #specify the data type with dtype parameter
    dtype=tf.float16
)

print('Matrix 2: ', another_matrix)
print('Matrix 2: ', another_matrix.ndim)

# Lets create a tensor
tensor = tf.constant(
    [
        [
            [1, 2, 3],
            [4, 5, 6]
        ],
        [
            [7, 8, 9],
            [10, 11, 12]
        ],
        [
            [13, 14, 15],
            [16, 17, 18]
        ]
    ]
)
print('Tensor: ', tensor)
print('Tensor Dimensions: ', tensor.ndim)

# Creating tensors with tf.Variable
# Create the same tensor with tf.Varaiable as above
changeable_tensor = tf.Variable([10,7])
unchangeable_tensor = tf.constant([10, 7])

print(changeable_tensor, unchangeable_tensor)

# Lets try change the elements in our changeable tensor
changeable_tensor[0].assign(7)
print('Changeable Tensor: ', changeable_tensor)

# Creating random tensors
# Create two random (but the same) tensors
# set seed for reproducibility
random_1 = tf.random.Generator.from_seed(42)
random_1 = random_1.normal(shape=(3, 2))
print('Random 1: ', random_1)

random_2 = tf.random.Generator.from_seed(42)
random_2 = random_2.normal(shape=(3, 2))
print('Random 2: ', random_2)

print(random_1 == random_2)

# Shuffle the order of elements in a tensor
# Shuffle a tensor (valuable for when you want to shuffle your data so the inherent order doesn't affect learning)
not_shuffled = tf.constant(
    [
        [10, 7],
        [3, 4],
        [2, 5]
    ]
)
print('Tensor not shuffled: ', not_shuffled)

# Shuffle our non-shuffled tensor
tf.random.set_seed(42)
shuffled = tf.random.shuffle(not_shuffled, seed=42)
print('Tensor shuffled: ', shuffled)

# Other ways to make tensors
# Create a tensor of all ones
tensor_of_ones = tf.ones([10, 7])
print('Tensor of ones: ', tensor_of_ones)

tensor_of_zeroes = tf.zeros(shape=(3, 4))
print('Tensor of Zeroes: ', tensor_of_zeroes)

# You can also turn NumOy arrays into tensors
import numpy as np
#create a NumPy array between 1 and 25
numpy_A = np.arange(1, 25, dtype=np.int32)
print('NumPy array: ', numpy_A)

# capital for matrix or tensor
# X = tf.constant(some_matrix)
# non-capital for vector
# y = tf.constant(vector)

A = tf.constant(numpy_A, shape=(2, 3, 4))
print('A: ', A)

B = tf.constant(numpy_A)
print('B: ', B)

# Getting information from tensors
# Create a rank 4 tensor (4 dimensions)
rank_4_tensor = tf.zeros(shape=(2, 3, 4, 5))
print('Rank 4 Tensor: ', rank_4_tensor)
print('Rank 4 Tensor Shape: ', rank_4_tensor.shape)
print('Rank 4 Tensor Dimension: ', rank_4_tensor.ndim)
print('Rank 4 Tensor Size: ', tf.size(rank_4_tensor))

# Get various attributes of our tensor
print('Datatype of every element: ', rank_4_tensor.dtype)
print('Number of dimensions (rank): ', rank_4_tensor.ndim)
print('Shape of tensor: ', rank_4_tensor.shape)
print('Elements along the 0 axis: ', rank_4_tensor.shape[0])
print('Elements along the  last axis: ', rank_4_tensor.shape[-1])
print('Total number of elements in our tensor: ', tf.size(rank_4_tensor))
print('Total number of elements in our tensor: ', tf.size(rank_4_tensor).numpy())

# Indexing tensors
# Get the first two elements of each dimensions
print('first two elements: ', rank_4_tensor[:2, :2, :2, :2])

# Get the first element from each dimension from each index except the final one
print('first element from each dimesion except final: ', rank_4_tensor[:1, :1, :1, :])

# Create arank 2 tensor (2 dimensions)
rank_2_tensor = tf.constant(
    [
        [10, 7],
        [3, 4]
    ]
)
print('Rank 2 Tensor Shape: ', rank_2_tensor.shape)
print('Rank 2 Tensor Dimesion: ', rank_2_tensor.ndim)

# Get the last item of each row of or rank 2 tensor
print('last item of each row: ', rank_2_tensor[:, -1])

# Add in extra dimensions to our rank 2 tensor
rank_3_tensor = rank_2_tensor[..., tf.newaxis]
print('Rank 3 Tensor: ', rank_3_tensor)

# Alternative to tf.newaxis
# '-1' means expand the final axis
alt_rank_3_tensor = tf.expand_dims(rank_2_tensor, axis=-1)
print('Alternative Rank 3 Tensor: ', alt_rank_3_tensor)

# Expand 0 axis
alt_rank_3_tensor = tf.expand_dims(rank_2_tensor, axis=0)
print('Alternative Rank 3 Tensor: ', alt_rank_3_tensor)

# Manipulating tensors (tensors operations)
# You can add values to a tensor using the addition operator
tensor = tf.constant(
    [
        [10, 7],
        [3, 4]
    ]
)
print('Tensor Addition: ', tensor + 10)

# Original tensor is unchanged
print('Tensor: ', tensor)

# Multiplication
print('Tensor Multiplication: ', tensor * 10)

# Substraction
print('Tensor Substraction: ', tensor - 10)

# We can use the tensorflow built-in tensorflow functions
print('Tensor tf Multiplication: ', tf.multiply(tensor, 10))

# Matrix multiplication in tensorflow
print('Tensor X Tensor: ', tf.matmul(tensor, tensor))
print('Tensor * Tensor: ', tensor * tensor)

# Matrix multiplication with python operator '@'
print('Tensor X Tensor: ', tensor @ tensor)

# Excercise recreate Tensor and matrix multiplication
A = tf.constant(
    [
        [1, 2, 5],
        [7, 2, 1],
        [3, 3, 3]
    ]
)
B = tf.constant(
    [
        [3, 5],
        [6, 7],
        [1, 8]
    ]
)
print('A X B: ', tf.matmul(A, B))

# Create two (3, 2) tensors
X = tf.constant(
    [
        [1, 2],
        [3, 4],
        [5, 6]
    ]
)

Y = tf.constant(
    [
        [7, 8],
        [9, 10],
        [11, 12]
    ]
)

# Lets change the shape of Y
Y2 = tf.reshape(Y, shape=(2, 3))
print('Y2: ', Y2)
print('X @ Y2: ', X @ Y2)

# Lets change shape of X
X2 = tf.reshape(X, shape=(2, 3))
print('X2: ', X2)
print('X2 @ Y: ', tf.matmul(X2, Y))

#Lets transpose Y or X
print('X @ Y.T: ', tf.matmul(X, tf.transpose(Y)))
print('X.T @ Y: ', tf.matmul(tf.transpose(X), Y))

# Perfrom the dot product on X and Y (requires X or Y to be transposed)
print('X.T @ Y: ', tf.tensordot(tf.transpose(X), Y, axes=1))

# Perform matrix multiplication between X and Y(transposed)
print('X @ Y.T: ', tf.matmul(X, tf.transpose(Y)))

# Check the values of Y, reshape Y and transposed Y
print('Normal Y: ')
print(Y, '\n')

print('Y reshaped to (2, 3): ')
print(tf.reshape(Y, shape=(2, 3)), '\n')

print('Y transposed: ')
print(tf.transpose(Y))

# Create a new tensor with default datatype (float32)
B = tf.constant(
    [1.7, 7.4]
)
print('B Datatype: ', B.dtype)

C = tf. constant(
    [7, 10]
)
print('C Datatype: ', C.dtype)

# Change from float32 to float16 (reduced precision)
B = tf.cast(B, dtype=tf.float16)
print('B reduced precision: ', B)

# Aggregating tensors
# Get the absolute values
D = tf.constant(
    [-7, -10]
)
print('D: ', D)
print('D absolute: ', tf.abs(D))

Z = tf.random.Generator.from_seed(24)
Z = tf.random.uniform(shape=(5, 3))
print('Z: ', Z)
# Get minimum of tensor
print('Minimum of Z: ', tf.reduce_min(Z))
# Get maximum of tensor
print('Maximum of Z: ', tf.reduce_max(Z))
# Get mean of a tensor
print('Mean of Z: ', tf.reduce_mean(Z))
# Get sum of a tensor
print('Sum of Z: ', tf.reduce_sum(Z))
# Get varaince of a tensor
print('Variance of Z: ', tf.math.reduce_variance(Z))
print('Variance of Z: ', tfp.stats.variance(Z))
# Get standard deviation of a tensor
print('Standard Deviation of Z: ', tf.math.reduce_std(Z))

# Find the positional maximum and minimum
print('Positional Maximum of Z axis 0: ', tf.argmax(Z, 0))
print('Positional Maximum of Z axis 1: ', tf.argmax(Z, 1))
print('Positional Minimum of Z axis 0: ', tf.argmin(Z, 0))
print('Positional Minimum of Z axis 1: ', tf.argmin(Z, 1))

# Squeezing a tensor (removing all single dimensions)
G = tf.constant(tf.random.uniform(shape=[50]), shape=(1, 1, 1, 1, 50))
print('G: ', G)
print('G Shape: ', G.shape)

G_squeezed = tf.squeeze(G)
print('G squeezed: ', G_squeezed)
print('G squeezed Shape: ', G_squeezed.shape)

# One-hot encoding tensors
# Create a list of indices (could be red, green, blue, or purple)
some_list = [0, 1, 2, 3]

# One hot encode encode our list of indices
one_hot = tf.one_hot(some_list, depth=5)
print('One-hot: ', one_hot)

one_hot_2 = tf.one_hot(some_list, depth=4, on_value='I love deep learning', off_value='I also like to code')
print('One-hot 2: ', one_hot_2)

# Squaring, log ,square root
# Create a new tensor
H = tf.range(1, 10)
print('H: ', H)
# Square
print('H**2: ', tf.square(H))

# Find the square root
print('H sqrt: ', tf.sqrt(tf.cast(H, dtype=tf.float32)))

# Find the log
print('H log: ', tf.math.log(tf.cast(H, dtype=tf.float32)))

# Tensors and NumPy
# Create a tensor directly from NumPy array
J = tf.constant(np.array([3., 7., 10.]))
print('J: ', J)

# Convert our tensor back to a NumPy array
print('NumPy Array: ', np.array(J))
print('NumPy Array Type: ', type(np.array(J)))

# Convert tensor J to a NumPy array
print('NumPy Array: ', J.numpy())
print('NumPy Array Type: ', type(J.numpy()))

# The default types of each are slighly different
numpy_J = tf.constant(np.array([3., 7., 10.]))
tensor_J = tf.constant([3., 7., 10.])
# Check the datatypes of each
print('NumPy_J Type: ', numpy_J.dtype)
print('Tensor_J Type: ', tensor_J.dtype)

# Excercise
print()
print(20 * '-' + 'Excercise' + 20 * '-')
print()
# 1. Create a vector, scalar, matrix and tensor with values of your choosing using tf.constant().
s_ex = tf.constant(24)
v_ex = tf.constant(
    [13, 24, 18]
)
M_ex = tf.constant(
    [
        [13, 24, 18],
        [10, 17, 97]
    ]
)

# 2. Find the shape, rank and size of the tensors you created in 1.
print('Scalar Excercise: ', s_ex)
print('Vector Excercise: ', v_ex)
print('Matrix Excercise: ', M_ex)

print('Scalar Excercise Shape: ', s_ex.shape)
print('Vector Excercise Shape: ', v_ex.shape)
print('Matrix Excercise Shape: ', M_ex.shape)

print('Scalar Excercise Rank: ', s_ex.ndim)
print('Vector Excercise Rank: ', v_ex.ndim)
print('Matrix Excercise Rank: ', M_ex.ndim)

print('Scalar Excercise Size: ', tf.size(s_ex))
print('Vector Excercise Size: ', tf.size(v_ex))
print('Matrix Excercise Size: ', tf.size(M_ex))

# 3. Create two tensors containing random values between 0 and 1 with shape [5, 300].
Tensor_1_ex = tf.random.set_seed(24)
Tensor_1_ex = tf.random.uniform(shape=(5, 300))

print('TEnsor 1 Excercise: ', Tensor_1_ex)
print('Check if numbers in Tensor are between 0 and 1')
print('Smallest : ', tf.reduce_min(Tensor_1_ex))
print('Biggest : ', tf.reduce_max(Tensor_1_ex))

Tensor_2_ex = tf.random.set_seed(13)
Tensor_2_ex = tf.random.uniform(shape=(5, 300))

print('Tensor 2 Excercise: ', Tensor_2_ex)
print('Check if numbers in Tensor are between 0 and 1')
print('Smallest : ', tf.reduce_min(Tensor_2_ex))
print('Biggest : ', tf.reduce_max(Tensor_2_ex))

# Multiply the two tensors you created in 3 using matrix multiplictaion.
print('Tensor 1 @ Tensor 2: ', tf.matmul(Tensor_1_ex, tf.transpose(Tensor_2_ex)))

# Multiply  the two tensors you created in 3 using dot product.
print('Tensor 1 dot Tensor 2: ', tf.tensordot(Tensor_1_ex, tf.transpose(Tensor_2_ex), axes=1))
print('Shape: ', tf.tensordot(Tensor_1_ex, tf.transpose(Tensor_2_ex), axes=0).shape)

# Create a tensor with random values between 0 and 1 with shape [224, 224, 3].
Tensor_3_ex = tf.random.set_seed(17)
Tensor_3_ex = tf.random.uniform(shape=(224, 224, 3))
print('Tensor 3 Excercise: ', Tensor_3_ex)

# Find the min and max values of the tensor you created in 6 along the first axis.
print('Tensor 3 Excercise Max: ', tf.reduce_max(Tensor_3_ex, axis=0))
print('Tensor 3 Excercise Min: ', tf.reduce_min(Tensor_3_ex, axis=0))

# Create a tensor with random values of shape [1, 224, 224, 3] then squeeze it to chage the shape to [224, 224, 3].
Tensor_4_ex = tf.random.set_seed(18)
Tensor_4_ex = tf.random.normal(shape=(1, 224, 224, 3))

print('Tensor 4 Excercise: ', Tensor_4_ex)
print('Tensor 4 Excercise squeezed: ', tf.squeeze(Tensor_4_ex))
print('Tensor 4 Excercise squeezed Shape: ', tf.squeeze(Tensor_4_ex).shape)

# Create a tensor with shape [10] using your own choice of values, then find the index which has the maximum value.
Tensor_5_ex = tf.constant(
    [13, 28, 17, 10, 97, 24, 7, 99, 3, 2]
)
print('Tensor 5 Excercise: ', Tensor_5_ex)
print('Tensor 5 Excercise Max Argument', tf.argmax(Tensor_5_ex))

# One-hot encode the tensor you created in 9.
one_hot_ex = tf.one_hot(Tensor_5_ex % 10, depth=10)
print('One-hot Excercise: ', one_hot_ex)

