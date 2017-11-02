import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Flatten, Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop, Adadelta
from keras.utils import np_utils, plot_model

import matplotlib as mpl
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import sys
sys.path.append('../utils')
from custom_activations import Itachi

mpl.rc('font', family = 'serif', size = 17)
mpl.rcParams['xtick.major.size'] = 5
mpl.rcParams['xtick.minor.size'] = 2
mpl.rcParams['ytick.major.size'] = 5
mpl.rcParams['ytick.minor.size'] = 2

def main():
    '''
    Using CNN to do image classification
    '''
    ### Basic parameters
    img_r = img_c = 28
    n_cls = 10
    
    ### Loading data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    n_train = len(x_train)
    n_test = len(x_test)
    y_train = np_utils.to_categorical(y_train, n_cls)
    y_test = np_utils.to_categorical(y_test, n_cls)
    ### We no longer need to flat the original image data
    x_train = x_train.reshape(n_train, img_r, img_c, 1).astype('float32')/255.
    x_test = x_test.reshape(n_test, img_r, img_c, 1).astype('float32')/255.
    
    ### Construct the model
    model = Sequential()
    model.add(Conv2D(32, kernel_size = (4, 4), input_shape = (img_r, img_c, 1),
    padding = 'same'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Activation(Itachi()))
    model.add(Conv2D(64, kernel_size = (4, 4), padding = 'same'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Activation(Itachi()))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation(Itachi()))
    model.add(Dropout(0.25))
    model.add(Dense(n_cls, activation = 'softmax'))
    model.summary()
    
    ### Compile the model
    optimizer = Adadelta()
    loss = 'categorical_crossentropy'
    metrics = ['accuracy']
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    ### Train the model
    res = model.fit(x_train, y_train, batch_size = 128, epochs = 6,
    verbose = 1, validation_data = (x_test, y_test))
    
    score, accu = model.evaluate(x_test, y_test, batch_size = 128, verbose = 1)
    print "\n\n"
    print "The score of the model is: ", score
    print "The accuracy for the test cases is: ",  accu
    
    ### Plot weights
    w1, b1, w2, b2, w3, b3, w4, b4 = model.get_weights()
    
    ## First cnnlayer
    print np.shape(w1)
    r = 4
    c = 8
    fig, axes = plt.subplots(r, c, figsize = (10, 5))
    for i, ax in enumerate(axes.flat):
        weights = w1[:, :, 0, i]
        ax.pcolor(weights)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect(1.)
    plt.savefig('CNLayer#1.png')
    
    
    ## Second cnnlayer
    print np.shape(w2)
    r = 4
    c = 4
    fig, axes = plt.subplots(r, c, figsize = (12, 6))
    for i, ax in enumerate(axes.flat):
        weights = w2[i/c, i%c, :, :]
        ax.pcolor(weights)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect(1.)
    plt.savefig('CNLayer#2.png')
    
    
    ## First fclayer
    print np.shape(w3)
    r = 8
    c = 16
    fig, axes = plt.subplots(r, c, figsize = (12, 7))
    for i, ax in enumerate(axes.flat):
        weights = w3[:, i]
        ax.pcolor(weights.reshape(64, 49))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect(1.)
    plt.savefig('FCLayer#1.png')
    
    ## Second fclayer
    print np.shape(w4)
    r = 5
    c = 2
    fig, axes = plt.subplots(r, c, figsize = (8, 10))
    for i,ax in enumerate(axes.flat[: r*c]):
        weights = w4[:, i]
        ax.pcolor(weights.reshape(8, 16))
        ax.set_aspect(1.)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.savefig('FCLayer#2.png')
    
    ### Show model structures
    plot_model(model, 'cnn.png')

if __name__ == '__main__':
    main()