---
layout: post
title: "Residual Learning with Stochastic_Depth"
tags:
    - python
    - notebook
---
# Residual Learning and Stochastic Depth

Residual networks are famed for receiving first place in latest ILSVRC image
classification.

They are able to achieve a start of the art performance in image classification
tasks beating previous VGG net
Stochastic Depth is a ridiculously simple idea which can help in training the
network even if the maximum depth is in order of
1000s.(http://arxiv.org/pdf/1603.09382v2.pdf). Total Code is available at
https://github.com/pradyu1993/Residual-Learning-and-Stochastic-Depth.

## Usual Importing

Here I am specifically using theano instead of tensorflow because for some
reason tensorflow performance is way less than theano for this code during
initial testing and tensorflow occupies whole vram while compiling the network.

**In [1]:**

{% highlight python %}
import os
os.environ['KERAS_BACKEND'] = 'theano'
os.environ['THEANO_FLAGS']='mode=FAST_RUN,device=gpu0,floatX=float32,optimizer=None'

import numpy as np
import collections

from keras.models import Model
from keras.optimizers import SGD
import keras.backend as K
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.callbacks import Callback, LearningRateScheduler
from keras.regularizers import l2
from keras.layers import Input, Activation, merge, Dense, Flatten, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D, AveragePooling2D

from resnet import bn_relu_conv, conv_bn_relu, residual_block

{% endhighlight %}

    Using Theano backend.
    Using gpu device 0: Tesla K80 (CNMeM is disabled, cuDNN 4007)


## Network Specific constants

**In [2]:**

{% highlight python %}
# constants
learning_rate = 0.1
momentum = 0.9
img_rows, img_cols = 32, 32
img_channels = 3
nb_epochs = 400
batch_size = 300
nb_classes = 10
pL = 0.5
weight_decay = 1e-4

# data
(X_train, Y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train.astype('float32')
img_gen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    horizontal_flip=True)
img_gen.fit(X_train)
Y_train = np_utils.to_categorical(Y_train, nb_classes)
{% endhighlight %}

# Building Blocks of Whole Network


<img src="http://i0.wp.com/deliprao.com/wp-content/uploads/2016/04/Screen-Shot-2016-04-01-at-1.33.56-PM.png">

The lamdba function in _shortcut function is what makes the network
**stochastic**. Effectively it is a implementation of the the picture shown
above.


**In [3]:**

{% highlight python %}
def _bottleneck(input, nb_filters, init_subsample=(1, 1)):
    conv_1_1 = bn_relu_conv(input, nb_filters, 3, 3, W_regularizer=l2(weight_decay), subsample=init_subsample)
    conv_3_3 = bn_relu_conv(conv_1_1, nb_filters, 3, 3, W_regularizer=l2(weight_decay))
    return _shortcut(input, conv_3_3)


def _shortcut(input, residual):
    stride_width = input._keras_shape[2] / residual._keras_shape[2]
    stride_height = input._keras_shape[3] / residual._keras_shape[3]
    equal_channels = residual._keras_shape[1] == input._keras_shape[1]

    shortcut = input
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Convolution2D(nb_filter=residual._keras_shape[1], nb_row=1, nb_col=1,
                                 subsample=(stride_width, stride_height),
                                 init="he_normal", border_mode="valid", W_regularizer=l2(weight_decay))(input)
        shortcut = Activation("relu")(shortcut)

    M1 = merge([shortcut, residual], mode="sum")
    M1 = Activation("relu")(M1)

    gate = K.variable(0.0, dtype="uint8")
    decay_rate = 1
    name = 'residual_'+str(len(gates)+1)
    gates[name]=[decay_rate, gate]
    return Lambda(lambda outputs: K.switch(gate, outputs[0], outputs[1]),
                  output_shape= lambda x: x[0], name=name)([shortcut, M1])

{% endhighlight %}

# Resnet

Residual Network with depth 110 is built with basic blocks define above.

**In [4]:**

{% highlight python %}
# http://arxiv.org/pdf/1512.03385v1.pdf
# 110 Layer resnet
def resnet():
    input = Input(shape=(img_channels, img_rows, img_cols))

    conv1 = conv_bn_relu(input, nb_filter=16, nb_row=3, nb_col=3, W_regularizer=l2(weight_decay))

    # Build residual blocks..
    block_fn = _bottleneck
    block1 = residual_block(conv1, block_fn, nb_filters=16, repetations=18, is_first_layer=True)
    block2 = residual_block(block1, block_fn, nb_filters=32, repetations=18)
    block3 = residual_block(block2, block_fn, nb_filters=64, repetations=18, subsample=True)

    # Classifier block
    pool2 = AveragePooling2D(pool_size=(8, 8))(block3)
    flatten1 = Flatten()(pool2)
    final = Dense(output_dim=10, init="he_normal", activation="softmax", W_regularizer=l2(weight_decay))(flatten1)

    model = Model(input=input, output=final)
    return model
{% endhighlight %}

# Callback and Linear Decay

**set_decay_rate**: This function sets decay rate of a gate in linear fashion.


**scheduler**: Function used to change learning rate after 25% and 50% of total
epochs are done.


**Gates_Callback**: Callback to change the state of gates after every batch.


**In [5]:**

{% highlight python %}
def set_decay_rate():
    for index, key in enumerate(gates):
        gates[key][0] = 1.0 - float(index)*pL / len(gates)

# Callbacks for updating gates and learning rate
def scheduler(epoch):
    if epoch < nb_epochs/4:
        return learning_rate
    elif epoch < nb_epochs/2:
        return learning_rate*0.1
    return learning_rate*0.01


class Gates_Callback(Callback):
    def on_batch_begin(self, batch, logs={}):
        num = 0
        probs = np.random.uniform(size=len(gates))
        for i,j in zip(gates, probs):
            if j > gates[i][0]:
                num = num + 1
                K.set_value(gates[i][1], 1)
            else:
                K.set_value(gates[i][1], 0)
        print num

    def on_train_end(self, logs={}):
        for i in gates:
            K.set_value(gates[i][1],1)
{% endhighlight %}

### Compiling the network

**In [6]:**

{% highlight python %}
gates=collections.OrderedDict()
model = resnet()
set_decay_rate()
model.compile(optimizer="adadelta", loss="categorical_crossentropy",metrics=["accuracy"])
{% endhighlight %}

# Visualizing the Network
Didnt run this snippet because the network is too big for a webpage, can be used
for debugging purposes.

**In [7]:**

{% highlight python %}
from IPython.display import SVG
from keras.utils.visualize_util import model_to_dot

SVG(model_to_dot(model,show_shapes=True).create(prog='dot', format='svg'))
{% endhighlight %}

# Training the Network
Training the network with previously coded Callbacks :)

**In [8]:**

{% highlight python %}
model.fit_generator(img_gen.flow(X_train, Y_train, batch_size=batch_size, shuffle=True),
                    samples_per_epoch=len(X_train),
                    nb_epoch=nb_epochs,
                    callbacks=[Gates_Callback(), LearningRateScheduler(scheduler)])
{% endhighlight %}
