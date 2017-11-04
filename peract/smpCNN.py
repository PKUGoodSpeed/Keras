import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop, Adadelta
from keras.utils import np_utils, plot_model
from sys import stdout, path
path.append('../utils')
from progress import ProgressBar
from custom_activations import Itachi
pbar = ProgressBar()

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('font', family = 'serif', size = 17)
mpl.rcParams['xtick.major.size'] = 5
mpl.rcParams['xtick.minor.size'] = 2
mpl.rcParams['ytick.major.size'] = 5
mpl.rcParams['ytick.minor.size'] = 2

def getMat(arr, win):
    '''
    Try to build CNN windows
    '''
    ans = []
    N = len(arr) - win + 1
    pbar.setBar(N)
    for i in range(len(arr) - win + 1):
        pbar.show(i)
        ans.append(np.array([np.array(vec) for vec in arr[i:i+win]]))
    return ans


def comp_cls_wts(y, pwr = 0.75):
    '''
    Used to compute class weights
    '''
    dic = {}
    for x in set(y):
        dic[x] = len(y)**pwr/list(y).count(x)**pwr
    return dic
    
def load_data(filename = 'data/train.txt', win = 16):
    '''
    Loading feature and action data from the file
    '''
    print "Loading the feature and action data!"
    fp = open(filename,'r')
    raw_x = []
    raw_y = []
    for line in fp:
        tmp = list(map(float, line.split(' ')))
        a = tmp[-1]
        raw_x.append(np.array(tmp[:-1] + [0., 0., 0.]))
        if(a > 0.5):
            raw_y.append(2)
        elif(a < -0.5):
            raw_y.append(1)
        else:
            raw_y.append(0)
    fp.close()
    print "Preprocess the data!"
    raw_x = getMat(raw_x , win)
    raw_y = raw_y[win-1:]
    print np.shape(raw_x), np.shape(raw_y)
    return raw_x, raw_y
    
def getTrainTest(raw_x, raw_y, ratio = 0.7):
    '''
    Separate the data into a train set and a test set
    '''
    assert len(raw_x) == len(raw_y)
    N_train = int(len(raw_x) * ratio)
    return raw_x[:N_train], raw_y[:N_train], raw_x[N_train:], raw_y[N_train:]
    
def main():
    
    ## Loading data
    n_cls = 3
    win = 16
    raw_x, raw_y = load_data(win = win)
    x_train, y_train, x_test, y_test = getTrainTest(raw_x, raw_y)
    cls_wts = comp_cls_wts(y_train, pwr = 0.64)
    print 1.*y_train.count(0)/len(y_train), 1.*y_test.count(0)/len(y_test)
    
    ## reshaping
    x_train = np.array(x_train).reshape(len(x_train), win, win, 1)
    x_test = np.array(x_test).reshape(len(x_test), win, win, 1)
    y_train = np_utils.to_categorical(np.array(y_train), n_cls)
    y_test = np_utils.to_categorical(np.array(y_test), n_cls)
    
    ## Construct the model
    model = Sequential()
    model.add(Conv2D(16, kernel_size = (4, 4), input_shape = (win, win, 1), padding = 'same'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Activation('relu'))
    model.add(Conv2D(16, kernel_size = (4, 4), padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Activation(Itachi()))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_cls, activation = 'softmax'))
    model.summary()
    
    ## Compile the model
    optimizer = SGD(lr=0.02)
    loss = 'categorical_crossentropy'
    metrics = ['accuracy']
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    ## Train the model
    res = model.fit(x_train, y_train, batch_size=512, epochs = 15, verbose = 1, validation_data=(x_test, y_test), 
    class_weight = cls_wts)
    loss, accu = model.evaluate(x_test, y_test, batch_size = 128)
    print "\n\n The loss of the model is: ", loss
    print "The accuracy of the model is: ", accu
    
    print "\n\n"
    y_pred_cls = model.predict_classes(x_test)
    print y_pred_cls
    print "# of 0: ", list(y_pred_cls).count(0)
    print "# of 1: ", list(y_pred_cls).count(1)
    print "# of 2: ", list(y_pred_cls).count(2)
    
    
if __name__ == '__main__':
    main()