# Image-Classification with ConvMixer
## Team
Zhaoyu Yin    zyyin@bu.edu <br>
Jiaye Liu     jiayel@bu.edu <br>
Junyang Li    jl981215@bu.edu

## Running the Script
All datasets can be imported from keras. <br>
TensorFlow >= 2.4.0 <br>
Require TensorFlow Addons <br>
pip install -U tensorflow-addons <br>

Recommend using SCC (2 cores 1 gpu) Then upload ipynb file.

## Goal
1. Understand the components and features of Vision Transformer
2. The implementation of the ConvMixer

## ViT Architecture
![overview](/examples/ViT_process.gif)
Linear projection of Flatten Patches <br>
Transformer Encoder <br>
MLP Head <br>

### Linear projection of Flatten Patches
![overview](/examples/ViT%20patch%20image.png)

As can be seen here, by using a convolutional layer, the initial image has been divided into several patches. The kernel size and the stride are the same as the patch size. Then, we can map each patch to one-dimensional vector, which is named as token. Also, it is concatenated with another class token for classification. The position embedding is the same as position encoding in the natural language processing. It can mark the position of a token in the sequence.

### Detailed flow chart of ViT
![overview](/examples/ViT_16.png)

This is the detailed flow chart of the vision transformer. We just showed the patch embedding. The encoder block iterates for a few times. Here is the detailed structure of the encoder block. Multi-head attention is an important mechanism. It can divide the model into multiple heads to create multiple subspaces so that the model can focus on different aspects of the information. It is great, but the computing cost is expensive. Then, it goes to the MLP block, which consists of multiple layers with dropout and gelu function. We only want to keep the patch embedding and replace other parts with the convolution Mixer network.

## ConvMixer Architecture
![overview](/examples/ConvMixer%20structure.png)
The architecture consist of three components
### Patch embedding layer
Uses a single conv2d layer with kernel size = patch size, strides = patch size, and a non-linearity function to split a n*n image into features vector of shape h x n/p x n/p.
This is then followed by a GELU activation function and a batch normalization post activation.
### ConvMixer block
First store the residual function from the previous loop to a temp value, then feed the patch into a depthwise convolution followed by a GELU and BN. After that, perform an addition between the layer input and the temp value. Lastly, feed the value into a pointwise convolution followed by a GELU and BN. Loop x times.
### Classification
After x loop, the residual function is feed into a global average pooling to flatten the value, then a dropout is applied followed by a softmax classifier to produce the inference.


## Experiment Setup
Use the CIFAR10 and CIFAR100 datasets and augment the data by keras layers
learning rate = 0.001
weight decay = 0.0001
Then split the dataset into train, validation and test data.

## Training Details
Dropout rate = 0.3
Use AdamW Optimizer

## Results
![overview](/examples/10181660272552_.pic.jpg)


## Reference
### Paper Reference
Paper 1: Transformer for Image Recognition at Scale <br>
https://arxiv.org/pdf/2010.11929.pdf <br>
Alexey Dosovitskiy,Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer,
Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby

Paper 2: Patches are all you need? <br>
https://openreview.net/pdf?id=TVHS5Y4dNvM <br>
Anonymous authors

### Dataset Reference
Cifar10 with 10 classes, 60000 images, 32 x 32 pixels <br>
https://www.cs.toronto.edu/~kriz/cifar.html 

Cifar100 with 100 classes, 60000 images, 32 x 32 pixels <br>
https://www.cs.toronto.edu/~kriz/cifar.html

### Prototype Vision Transformer Code
Author: [Khalid Salama] <br>
Description: Implementing the Vision Transformer (ViT) model for image classification. <br>
https://github.com/keras-team/keras-io/blob/master/examples/vision/image_classification_with_vision_transformer.py
