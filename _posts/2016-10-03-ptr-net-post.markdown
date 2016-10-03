---
layout: post
title: "Pointer_Network"
tags:
    - python
    - notebook
---
# Pointer Network

#### This is a implemention of Pointer Networks in keras(similar to seq2set)
[http://arxiv.org/pdf/1511.06391v4.pdf](http://arxiv.org/pdf/1511.06391v4.pdf)

Below lines just do the required imports.

**In []:**

{% highlight python %}
import os
import cPickle
os.environ['KERAS_BACKEND'] = 'tensorflow'

import keras
from keras.engine.topology import Layer
from keras import initializations
from keras.layers.recurrent import time_distributed_dense
from keras.activations import tanh, softmax
from keras.layers import Input, LSTM, Dense, RepeatVector, Lambda, Activation
from keras.layers.wrappers import TimeDistributed
from keras.engine import InputSpec
from keras.models import Model
import keras.backend as K 
from keras.callbacks import Callback, LearningRateScheduler

import numpy as np
{% endhighlight %}

### PointerLSTM

This is a custom decoder layer that returns pointer for the input sequence. This
is a slight modification to the famous soft attention LSTM paper by Bandanau et
al( Neural machine translation by jointly learning to align and translate. In
Proc. ICLR, 2015).

#### A visualization of what the decoder is doing.

<img src ="https://raw.githubusercontent.com/pradyu1993/seq2set-keras/master/Ptr-net.png" width="65%">

**In []:**

{% highlight python %}
class PointerLSTM(LSTM):
    def __init__(self, hidden_shape, *args, **kwargs):
        self.hidden_shape = hidden_shape
        self.input_length=[]
        super(PointerLSTM, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        super(PointerLSTM, self).build(input_shape)        
        self.input_spec = [InputSpec(shape=input_shape)]
        init = initializations.get('orthogonal')
        self.W1 = init((self.hidden_shape, 1))
        self.W2 = init((self.hidden_shape, 1))
        self.vt = init((input_shape[1], 1))
        self.trainable_weights += [self.W1, self.W2, self.vt]

    def call(self, x, mask=None):
        input_shape = self.input_spec[0].shape
        en_seq = x
        x_input = x[:, input_shape[1]-1, :]
        x_input = K.repeat (x_input, input_shape[1])
        initial_states = self.get_initial_states(x_input)

        constants = super(PointerLSTM, self).get_constants(x_input)
        constants.append(en_seq)
        preprocessed_input = self.preprocess_input(x_input)

        last_output, outputs, states = K.rnn(self.step, preprocessed_input,
                                             initial_states,
                                             go_backwards=self.go_backwards,
                                             constants=constants,
                                             input_length=input_shape[1])

        return outputs

    def step(self, x_input, states):
        input_shape = self.input_spec[0].shape
        en_seq = states[-1]
        _, [h, c] = super(PointerLSTM, self).step(x_input, states[:-1])
        
        # vt*tanh(W1*e+W2*d)
        dec_seq = K.repeat(h, input_shape[1])
        Eij = time_distributed_dense(en_seq, self.W1, output_dim=1)
        Dij = time_distributed_dense(dec_seq, self.W2, output_dim=1)
        U = self.vt * tanh(Eij + Dij)
        U = K.squeeze(U, 2)
        
        # make probability tensor
        pointer = softmax(U)
        return pointer, [h, c]

    def get_output_shape_for(self, input_shape):
        # output shape is not affected by the attention component
        return (input_shape[0], input_shape[1], input_shape[1])
{% endhighlight %}

I prepared the data for travelling sales man problem using this algo
[https://gist.github.com/mlalevic/6222750](https://gist.github.com/mlalevic/6222750).
Generated around 10^5 examples as input dataset.

The output results though exciting werent very impressive. I am running more
epochs to see if that changes.

**In []:**

{% highlight python %}

f = open("data.pkl", 'rb')
X,Y = cPickle.load(f)

hidden_size = 512
seq_len = 11
nb_epochs = 100
learning_rate = 0.1

main_input = Input(shape=(seq_len, 2), name='main_input')

encoder = LSTM(output_dim = hidden_size, return_sequences = True, name="encoder")(main_input)
decoder = PointerLSTM(hidden_size, output_dim = hidden_size, name="decoder")(encoder)

model = Model(input=main_input, output=decoder)
model.compile(optimizer='adadelta',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X,Y, nb_epoch=nb_epochs, batch_size=8000,callbacks=[LearningRateScheduler(scheduler),])
model.save_weights('model_weight_100.hdf5')
{% endhighlight %}
