# PyTroch nn module

1. Building the neural network using nn module.
The torch.nn module in PyTorch is a core library that provides a wide array of classes and functions designed to help developers build neural networks efficiently and effectively. It abstracts the complexity of creating and training neural networks by offering pre-built layers, loss functions, activation functions, and other utilities, enabling you to focus on designing and experimenting with model architectures.  

Key Componenets of torch.nn

### Module Layers
- nn.Module: The base class for all neural network modules. Your custom models and layers should subclass this class.
- Common Layers: nn.Linear (fully connected layer), nn.Conv2D, nn.LSTM, and many more

### Activation Functions
Functions like nn.RELU, nn.Sigmoid, and nn.Tanh introduce non-linearities to the model, allowing it to learn complex patterns.

### Loss Functions
Provides loss function such as nn.CrossEntropyLoss, nn.MSELoss, and nn.NLLLoss to quntify the difference between the model's predictions and the actual targets.

### Container Modules
nn.Sequential: A sequential container to stack layers in order

### Regularization and Dropout:
Layers like nn.Dropout and nn.BatchNorm2D help preventing overfitting and improve the model's ability to generalize new data.



2. Using built-in activation function.

3. Using built-in loss function.

4. Using built-in optimizer.

