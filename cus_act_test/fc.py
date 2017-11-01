import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils, plot_model

import matplotlib as mpl
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import sys
sys.path.append('..')
from custom_activations import Itachi

mpl.rc('font',family = 'serif',size = 17)
mpl.rcParams['xtick.major.size'] = 5
mpl.rcParams['xtick.minor.size'] = 2
mpl.rcParams['ytick.major.size'] = 5
mpl.rcParams['ytick.minor.size'] = 2

def main():
    
    ### Gettning data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    n_train = len(x_train)
    n_test = len(x_test)
    n_cls = 10
    x_train = x_train.reshape(n_train, -1).astype('float32')/255.
    x_test = x_test.reshape(n_test, -1).astype('float32')/255.
    y_train = np_utils.to_categorical(y_train, n_cls)
    y_test = np_utils.to_categorical(y_test, n_cls)
    
    print np.shape(x_train), np.shape(y_train), np.shape(x_test), np.shape(y_test)
    
    ### Create model
    model = Sequential()
    model.add(Dense(128, input_shape = (28*28,)))
    model.add(Activation(Itachi()))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    model.summary()
    
    ### Compile the model
    optimizer = SGD()
    loss = 'categorical_crossentropy'
    metrics = ['accuracy']
    model.compile(loss = loss, optimizer = optimizer, metrics = metrics)
    
    ### Train the model
    res = model.fit(x_train, y_train, batch_size = 128, epochs = 5, shuffle = True, validation_data = (x_test, y_test))
    score, accu = model.evaluate(x_test, y_test, batch_size = 128)
    
    print '\n\n'
    print "The score of the model is ", score
    print "The accuracy for the testing data is ", accu
    
    ### Show network structure
    plot_model(model, 'fc.png')
    
main()