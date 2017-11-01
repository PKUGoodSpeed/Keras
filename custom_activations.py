'''
In this file, we implement several customized activation functions
'''
from keras import initializers, regularizers, constraints
from keras.engine import Layer, InputSpec
from keras import backend as K
from keras.legacy import interfaces
import tensorflow as tf

class Itachi(Layer):
    '''
    activation(x) := tanh(k * x)
    '''
    def __init__(self, coeff = 1., **kwargs):
        super(Itachi, self).__init__(**kwargs)
        self.supports_masking = True
        self.coeff = K.cast_to_floatx(coeff)
    
    def call(self, inputs):
        return tf.tanh(inputs*self.coeff)
        
    def get_config(self):
        config = {'coeff' : float(self.coeff)}
        base_config = super(Itachi, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))