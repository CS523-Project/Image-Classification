# Image-Classification with ConvMixer
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
