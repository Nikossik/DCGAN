# DCGAN

`DCGANs` use some of the basic principles of CNNs and thus have become one of the most widely used architectures in practice due to their fast convergence, and also due to the fact that they can be very easily adapted to more complex options (using labels as conditions, using residual blocks etc.). Some of the most important problems that `DCGANs` solve are:

- `D` is designed so that it mainly solves the problem of image classification under observation
- Filters learned by GAN can be used to draw specific objects in the generated image.
- `G` contains vectorized properties that can learn very complex semantic representations of objects.

Here are some basic guidelines to consider when creating a stable `DCGAN` versus the `CNN` standard:
- Replace pooling functions with sharp convolutions
- Use `BatchNorm`
- Avoid using fully connected hidden layers (not displayed).
- For `G` - use `ReLU` and `Tanh` activations to withdraw
- For `D` - use `LeakyReLU` activations (and sigmoid function for probabilistic output)


#### Here is the most standard structure of a DCGAN Generator:
![DCGAN](https://www.machinelearningmastery.ru/img/0-493475-599034.png)

### DCGAN generator structure
As we can see, its initial contribution is simply a `(1, 100) noise vector` that goes through `4 convolutional layers` with up-frequency and 2 steps to obtain the resulting `RGB image of size (64, 64, 3)`. To achieve this, the input vector is projected onto a `1024-dimensional output` signal to match the input of the first cone, which we will see later.

What would a standard discriminator look like? Well, about what you expect, let's see:

![out](https://www.machinelearningmastery.ru/img/0-455821-748638.png)

### DCGAN Discriminator Structure
This time we have an input image `(64, 64, 3)`, the same as output. We downsample this 4 standard `Conv layers` again in increments of 2. In the final output layer, the image is vector-aligned, which is usually fed into a `sigmoid function`, which then outputs a D Prediction for that image (one value representing the probability in the range `[0,1] - dog = 1 or no dog = 0`),

Well, now you have seen the basic idea of `Gans` and `DCGANs`. So now we can start creating some dogs using Tenserflow and Keras too :).
---
## Realisation of DCGAN on Keras:

### Data
We use `keras.preprocessing` and `matplotlib.pyplot` to output the data and prepare it.

```python
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
import os
```
First, let's print the first sample

```python
sample = image.load_img("faces/cropped/1.png")
plt.imshow(sample)
plt.title("Sample Image")
plt.axis('off')
plt.show()
```


