# What are Tensors
Tensor is a specialized multi-dimensional array designed for mathematical and computational efficiency.

> Dimension: Directions

1. Scalars: 0-Dimensional tensors(a single number)
Ex: Loss value, etc

2. Vectors: 1-dimensional tensors(a list of numbers)
Sequence of collection of values/scalars.
Ex: Array/Word Embedding [__, __, __]

3. Matrices: 2-dimensional tensors (2D Grid of numbers)
Tabular/Grid like data. Collection of vectors?

Ex: Grayscale Images

4. Three-dimensional Tensors: Colored images
Adds a third dimensional to 2-dim often used for stacking
Ex: RGB Images

5. 4D Tensors: Batches of RGB Images
Ex:

6. 5D Tensors: Video Data

## Why are tensors useful?

1. Mathematical operations
Enables efficient mathematical computations (additions, multiplication, dot product, etc) necessary for neural network operation.

2. Representation of real world data
data like image, audio, video, and text can be represented as a tensor

3. Efficient computations
Tensors are optimized for hardware acceleration, allowing computations on GPU or TPUs, which are crucial for training deep learning models.


## Where are tensors used in deep learning?
1. Data Storage
Training data (image, text,e tc) is store in tensors

2. Wights and biases
The learnable parameters of a neural network (weights, biases) are stored as tensors

3. Matric Operations
Neural networks invovle opeartions like matrix multiplication, dot product, and broad-casting

4. Training process

