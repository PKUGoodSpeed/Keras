import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils, plot_model

from IPython.display import SVG
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.switch_backend('agg')

mpl.rc('font',family = 'serif',size = 17)
mpl.rcParams['xtick.major.size'] = 5
mpl.rcParams['xtick.minor.size'] = 2
mpl.rcParams['ytick.major.size'] = 5
mpl.rcParams['ytick.minor.size'] = 2

def main():
    '''
    Simply use two fc_layers to train the most simple image classification problem 
    using mnist datasets
    '''
    
    ## loading data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    ## preprocess data
    n_train = np.shape(x_train)[0]
    n_test = np.shape(x_test)[0]
    n_cls = 10
    x_train = x_train.reshape(n_train, 28*28).astype('float32')/255.
    x_test = x_test.reshape(n_test, 28*28).astype('float32')/255.
    y_train = np_utils.to_categorical(y_train, n_cls)
    y_test = np_utils.to_categorical(y_test, n_cls)
    
    ## creat model
    model = Sequential()
    model.add(Dense(128, input_shape = (28*28,), activation = 'sigmoid', name = 'fc_1'))
    model.add(Dense(10, activation = 'softmax', name = 'fc_2'))
    
    ## show the layer summary
    model.summary()
    
    ## set optimizer
    sgd_opt = SGD()
    
    ## compile the model
    model.compile(loss = 'categorical_crossentropy', 
    optimizer = sgd_opt, metrics=['accuracy'])
    
    ## Train the model
    hist = model.fit(x_train, y_train,
    batch_size = 128, epochs = 5,
    verbose = 1, shuffle = True,
    validation_data=(x_test, y_test))
    
    ## Get results
    score, accu = model.evaluate(x_test, y_test, batch_size = 128)
    print("\nThe score of the model is: ", score)
    print("The accuracy for the test cases is: ", accu)
    
    w1, b1, w2, b2 = model.get_weights()
    print("Weight are in this shape: ", np.shape(w1), np.shape(b1), np.shape(w2), np.shape(b2))
    
    ## Visualize the weights
    ## Showing the first layer
    r, c = (8, 16)
    fig, axes = plt.subplots(r, c, figsize = (12, 6))
    for i,ax in enumerate(axes.flat):
        weights = w1[:, i]
        ax.pcolor(weights.reshape(28, 28))
        ax.set_aspect(1.)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.savefig('layer1.png')
    
    ## Showing the second layer
    r, c = (5, 2)
    fig, axes = plt.subplots(r, c, figsize = (8, 10))
    for i,ax in enumerate(axes.flat[: r*c]):
        weights = w2[:, i]
        ax.pcolor(weights.reshape(8, 16))
        ax.set_aspect(1.)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.savefig('layer2.png')
    
    ## Visualize the model configuration
    plot_model(model, to_file='model.png')
    
    

main()