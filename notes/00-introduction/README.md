# Introduction to PyTorch

2002: Torch Framework for tensor based operations also we can perform tensor based operations on GPUs
There were 2 problems with Torch:
    - It was lua based (lua: not a familiar programming): Learn lua, then torch, then create applciations
    Meta researchers identified this issue and introduced: PyTorch
    PyTroch: Marriage of Python and Torch 
    - the second issue with torch was the static computation graph
    

First version of pyTorch ___year___
1) python integration: dev can use torch using python also compatible with python libraries like 
2) Dynamic computation graph:
    - but what is computation graph?
    Visual way of representing mathematics
    Dynamic computation graph: Your neural net architecture can be changed in runtime


Secon release of pytorch:
    filled the gap between research and production
    They introduced torchscript for model serialization
    at the same time the facebook team was working another libray called caffe2 and in the 2018 release the also merged the caffe2 with pytorch

Subsequent 1.x version
    - distributed training
    - ONNX: For interoperatibility with other frameworks 
    - Quantization: stored model weights e.g., in float variables, quantization are technique for reducing the size of models | Model compresion
    - Torch Vision: For cv
    - TorchText: For nlp
    - torchaudio


PyTorch version 2.0
Worked on optimization:
    - latency 
    - througput: how much data you can process at a time


## Core features of pytorch
1. Tensor Computation
2. GPU Acceleration
3. Dynamic Computation Graph
4. Automatic Differentiation
5. Distributed Training
6. Interopeartibility with other libraries

## PyTorch vs. Tensorflow



One should learn both PyTroch and Tensorflow. If you are not left with enough time and want to dive in the field of GenAI then go with PyTorch.

## PyTorch Core Modules

| Module | Description|
| `torch`| 
| torch.autograd
torch.nn
torch.optim
torch.utils.data
torct.jit
torch.distributed
torch.cuda
torch.backends
torch.multiprocessing
torch.quantization
torch.onix

## PyTorch Domain Libraries
| Library | Description|
torchvision
torchtext(nowadays huggingface is used)
torchaudio| 
torcharrow | For structured/time series data
torchserve | For deployment
pytroch_lighting | Like high level API


## PyTorch Ecosystem libraries

| Library | Description|
Hugging face transformer | llma/genai 
fastai| simplifies pytorch code, high level lib
pytorch geometric  | Neural nets
TorchMetrics | for metrics 
TorchElastic| 
optuna | for hypterparameter tuning
catalyst
ignite
allen nlp | nlp related tasks
skorch| Marriage of scikit learn and pytorch 
PyTorch forecasting     
Tensorboard for Pytorch | for visualization


## Who Uses PyTorch
company | Products/servises | Description of usage
meta
ms(azure, bing, etc)
tesla (autopilot)
openai
uber