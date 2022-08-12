# Image-Classification with ConvMixer
## Team
Zhaoyu Yin    zyyin@bu.edu <br>
Jiaye Liu     jiayel@bu.edu <br>
Junyang Li    jl981215@bu.edu

## Architecture
![overview](/examples/ConvMixer%20structure.png)
The architecture consist of three components
### Patch embedding layer
Uses a single conv2d layer with kernel size = patch size, strides = patch size, and a non-linearity function to split a n*n image into features vector of shape h x n/p x n/p.
This is then followed by a GELU activation function and a batch normalization post activation.
### ConvMixer block
First store the residual function from the previous loop to a temp value, then feed the patch into a depthwise convolution followed by a GELU and BN. After that, perform an addition between the layer input and the temp value. Lastly, feed the value into a pointwise convolution followed by a GELU and BN. Loop x times.
### Classification
After x loop, the residual function is feed into a global average pooling to flatten the value, then a dropout is applied followed by a softmax classifier to produce the inference.


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
