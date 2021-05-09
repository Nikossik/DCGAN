# DCGAN
Making DCGAN on Keras

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

#### Here you can download a dataset: [kaggle](https://www.kaggle.com/kostastokis/simpsons-faces)
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
![1](/photo/1.png)

And now let's prepare all the data for training

```python
X_train = []
img_dir = 'faces/cropped'
for path in os.listdir(img_dir):
    img = image.load_img(os.path.join(img_dir, path))
    img = (image.img_to_array(img, dtype='float32')- 127.5)/127.5
    X_train.append(img)
X_train = np.array(X_train)
```
Also let's see the shape of `X_train`

```python
X_train.shape
```
```
(9877, 200, 200, 3)
```
We now have complete data for training.
---
### Creating Model
#### Some constants

- NO_OF_BATCHES - number of batches will pass through the discriminator in 1 epoch
- HALF_BATCH - Discriminator requires half fake and half real samples
- NOISE_DIM - Dimension of the RANDOM NOISE VECTOR
- Optimizer - [Adam](https://keras.io/api/optimizers/adam/)

```python
TOTAL_EPOCHS = 300
BATCH_SIZE = 256

NO_OF_BATCHES = int(X_train.shape[0]/BATCH_SIZE)

HALF_BATCH = int(BATCH_SIZE/2)

NOISE_DIM = 100

from keras.optimizers import Adam
adam = Adam(learning_rate=2e-4,beta_1=0.5)
```
--- 
### Gan Model

```python
from keras.layers import *
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
```
#### Generator
- Using `TRANSPOSE CONVOLUTION`
- We will use `Conv2DTranspose()` layer
- This layer also increases the channels as well as performs upsampling

Firtsly, we use upsampling of noise

```python
generator = Sequential()

generator.add(Dense(25*25*128, input_shape=(NOISE_DIM,)))
```
- Double Activation Size :: Upsampling ( 50 X 50 X 64 )
- strides is required as it leads to mapping
- Stride of 2 implies double the spatial arrangement
- `i.e (25, 25) ---> (50, 50)`

```python
generator.add(Reshape((25,25,128)))
generator.add(LeakyReLU(0.2))
generator.add(BatchNormalization())

generator.add(Conv2DTranspose(64, kernel_size=(5,5), strides=(2,2), padding='same'))
generator.add(LeakyReLU(0.2))
generator.add(BatchNormalization())
```
- i.e (50 X 50 X 64) ---> (100 X 100 X 32)

```python
generator.add(Conv2DTranspose(32, kernel_size=(5,5), strides=(2,2), padding='same'))
generator.add(LeakyReLU(0.2))
generator.add(BatchNormalization())
```
- Double Activation Size :: `Upsampling` ( 200 X 200 X 3 )

```python
generator.add(Conv2DTranspose(3, kernel_size=(5,5), padding='same', strides=(2,2), activation='tanh'))
```
Finally, we compile model and look at the summary:

```python
generator.compile(loss='binary_crossentropy', optimizer=adam)
generator.summary()
```
```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense (Dense)                (None, 80000)             8080000   
_________________________________________________________________
reshape (Reshape)            (None, 25, 25, 128)       0         
_________________________________________________________________
leaky_re_lu (LeakyReLU)      (None, 25, 25, 128)       0         
_________________________________________________________________
batch_normalization (BatchNo (None, 25, 25, 128)       512       
_________________________________________________________________
conv2d_transpose (Conv2DTran (None, 50, 50, 64)        204864    
_________________________________________________________________
leaky_re_lu_1 (LeakyReLU)    (None, 50, 50, 64)        0         
_________________________________________________________________
batch_normalization_1 (Batch (None, 50, 50, 64)        256       
_________________________________________________________________
conv2d_transpose_1 (Conv2DTr (None, 100, 100, 32)      51232     
_________________________________________________________________
leaky_re_lu_2 (LeakyReLU)    (None, 100, 100, 32)      0         
_________________________________________________________________
batch_normalization_2 (Batch (None, 100, 100, 32)      128       
_________________________________________________________________
conv2d_transpose_2 (Conv2DTr (None, 200, 200, 3)       2403      
=================================================================
Total params: 8,339,395
Trainable params: 8,338,947
Non-trainable params: 448
_________________________________________________________________
```
### Discriminator

- INPUT : 200 X 200 X 3

```python
discriminator = Sequential()

discriminator.add(Conv2D(32, (5,5), strides=(2,2), padding='same', input_shape=(200, 200, 3)))
discriminator.add(LeakyReLU(0.2))
```
- 100 X 100 X 32 ---> 50 X 50 X 64

```python
discriminator.add(Conv2D(64, (5,5), strides=(2,2), padding='same'))
discriminator.add(LeakyReLU(0.2))
```
- 50 X 50 X 64 ---> 25 X 25 X 128

```python
discriminator.add(Conv2D(128, (5,5), strides=(2,2), padding='same'))
discriminator.add(LeakyReLU(0.2))

discriminator.add(Flatten())
discriminator.add(Dense(8192))
discriminator.add(LeakyReLU(0.2))

discriminator.add(Dense(1, activation='sigmoid'))
```
Finally, we compile model and look at the summary
```python
discriminator.compile(loss='binary_crossentropy', optimizer=adam)

discriminator.summary()
````
```
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 100, 100, 32)      2432      
_________________________________________________________________
leaky_re_lu_3 (LeakyReLU)    (None, 100, 100, 32)      0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 50, 50, 64)        51264     
_________________________________________________________________
leaky_re_lu_4 (LeakyReLU)    (None, 50, 50, 64)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 25, 25, 128)       204928    
_________________________________________________________________
leaky_re_lu_5 (LeakyReLU)    (None, 25, 25, 128)       0         
_________________________________________________________________
flatten (Flatten)            (None, 80000)             0         
_________________________________________________________________
dense_1 (Dense)              (None, 8192)              655368192 
_________________________________________________________________
leaky_re_lu_6 (LeakyReLU)    (None, 8192)              0         
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 8193      
=================================================================
Total params: 655,635,009
Trainable params: 655,635,009
Non-trainable params: 0
_________________________________________________________________
```
---
### Combining to Make a GAN

We must freeze D and training G

```python
discriminator.trainable = False

gan_input = Input(shape=(NOISE_DIM,))
generated_img = generator(gan_input)
gan_output = discriminator(generated_img)
```
Combining the Model :: Functional API

```python
model = Model(gan_input, gan_output)
model.compile(loss='binary_crossentropy', optimizer=adam)
```
---
### Training GAN

- Step-1 is performed on the generator only
- Step-2 is performed on model ( which includes both generator and discriminator [FROZEN] )

```python
d_loss_list = []
g_loss_list = []

for epoch in range(TOTAL_EPOCHS):

    epoch_d_loss = 0.0
    epoch_g_loss = 0.0

    # mini-batch SGD

    for step in range(NO_OF_BATCHES):

        # Step-1 : Training Discriminator keeping Generator as Frozen
        # 50% real data + 50% fake data

        # Real Data
        idx = np.random.randint(0, X_train.shape[0], HALF_BATCH)
        real_imgs = X_train[idx]

        # Fake Data ( for HALF_BATCH number of examples )
        noise = np.random.normal(0,1,size=(HALF_BATCH, NOISE_DIM))
        fake_imgs = generator.predict(noise)

        # Labels
        real_y = np.ones((HALF_BATCH, 1))*0.9 # One Side Label Smoothing for discriminator
        fake_y = np.zeros((HALF_BATCH, 1))

        # Training our discriminator
        d_loss_real = discriminator.train_on_batch(real_imgs, real_y)
        d_loss_fake = discriminator.train_on_batch(fake_imgs, fake_y)

        # loss for the current batch
        d_loss = 0.5*d_loss_real + 0.5*d_loss_fake

        # This will get added in total_loss for the epoch
        epoch_d_loss += d_loss


        # Step-2 : Training Generator keeping Discriminator as Frozen

        noise = np.random.normal(0, 1, size=(BATCH_SIZE, NOISE_DIM))

        # All the fake images are treated as real
        ground_truth_y = np.ones((BATCH_SIZE,1))

        g_loss = model.train_on_batch(noise, ground_truth_y)

        epoch_g_loss += g_loss

    print("Epoch %d Discriminator Loss %.4f Generator Loss %.4f"%((epoch+1), epoch_d_loss/NO_OF_BATCHES, epoch_g_loss/NO_OF_BATCHES))

    d_loss_list.append(epoch_d_loss/NO_OF_BATCHES)
    g_loss_list.append(epoch_g_loss/NO_OF_BATCHES)

    if( epoch + 1) % 25 == 0:
        
        save_imgs(epoch + 1)
```
---
### Epoch 25:
![2](/photo/25.png)
### Epoch 50
![3](/photo/50.png)
### Epoch 100
![4](/photo/100.png)
### Epoch 170
![5](/photo/175.png)
### Epoch 250
![6](/photo/250.png)
### Epoch 300
![7](/photo/300.png)
