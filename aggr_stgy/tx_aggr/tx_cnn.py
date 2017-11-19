import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop, Adadelta
from keras.utils import np_utils, plot_model
import tensorflow as tf

from sys import stdout, path
path.append('../../utils')
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
plt.switch_backend('agg')

def getMat(arr, win = 25):
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


def comp_cls_wts(y, pwr = 0.64):
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
        tmp = np.array(map(float, line.split(' ')))
        a = tmp[-1]
        raw_x.append(tmp[:-1])
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
    return np.array(raw_x), np.array(raw_y)

def getTrainTest(raw_x, raw_y, n_train = 120000, n_test = 40000):
    '''
    Separate the data into a train set and a test set
    '''
    assert len(raw_x) == len(raw_y)
    assert n_train + n_test <= len(raw_y)
    return raw_x[:n_train], raw_y[:n_train], raw_x[n_train:n_train+n_test], raw_y[n_train:n_train+n_test]

def getPrice(filename = 'data/prcs.txt', win = 25, n_train = 120000, n_test = 40000):
    '''
    Getting the ask and bid prices for train and test sets
    '''
    fp = open(filename, 'r')
    prc = []
    for line in fp:
        prc.append(np.array(map(float, line.split(' '))))
    ask = prc[0][win-1:]
    bid = prc[1][win-1:]
    return ask[:n_train], bid[:n_train], ask[n_train:n_train+n_test], bid[n_train:n_train+n_test]


def getPnl(act, ask, bid, fee = 0.5):
    '''
    Given a particular action list, predict the number of flips and pnl 
    we can obtain from this action list
    '''
    pnl = 0.
    pos = 0.
    prc = 0.
    n = len(act)
    flip = 0
    for i in range(n):
        if act[i] == 2 and pos < 1: 
            pnl -= (ask[i] + fee/(1-pos))*(1-pos)
            prc = (ask[i] + fee/(1-pos))
            pos = 1
            flip += 1
        elif act[i] == 1 and pos > -1:
            pnl += (bid[i] - fee/(pos+1))*(pos+1)
            prc = (bid[i] - fee/(pos+1))
            pos = -1
            flip += 1
    pnl += pos*prc
    return flip, pnl
    
def main():

    ## Loading data
    n_cls = 3
    n_feat = 13
    win = 25
    n_train = 300000
    n_test = 100000
    raw_x, raw_y = load_data(win = win)
    x_train, y_train, x_test, y_test = getTrainTest(raw_x, raw_y,
    n_train = n_train, n_test = n_test)
    cls_wts = comp_cls_wts(y_train)
    
    ## Getting price data
    ask_train, bid_train, ask_test, bid_test = getPrice(win = win, 
    n_train = n_train, n_test = n_test)
    print np.shape(x_train), np.shape(y_train), np.shape(x_test) ,np.shape(y_test)
    print "Using perfect action"
    print "In sample performance: ", getPnl(y_train, ask_train, bid_train)
    print "Out of sample performance: ", getPnl(y_test, ask_test, bid_test)
    
    ## Reshape the date
    x_train = np.array(x_train).reshape(n_train, win, n_feat, 1)
    x_test = np.array(x_test).reshape(n_test, win, n_feat, 1)
    y_train = np_utils.to_categorical(y_train, 3)
    y_test = np_utils.to_categorical(y_test, 3)
    
    ## Construct CNN model
    model = Sequential()
    model.add(Conv2D(24, kernel_size = (5, 2*n_feat - 1), input_shape = (win, n_feat, 1), padding = 'same'))
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.1))
    model.add(Conv2D(48, kernel_size = (5, n_feat), padding='valid'))
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(n_cls, activation = 'softmax'))
    model.summary()
    
    ## Compile the model
    optimizer = SGD()
    loss = 'categorical_crossentropy'
    metrics = ['accuracy']
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    
    ## Run iterations to get convergence behavior
    steps = []
    train_loss = []
    test_loss = []
    train_accu = []
    test_accu = []
    in_pnl = []
    out_pnl = []
    in_flip = []
    out_flip = []
    for i in range(300):
	print i
        steps.append(i)
        res = model.fit(x_train, y_train, batch_size=512, epochs = 10, verbose = 0, 
        validation_data=(x_test, y_test), class_weight = cls_wts)
        lr, ar = model.evaluate(x_train, y_train, batch_size=128)
        train_loss.append(lr)
        train_accu.append(ar)
        le, ae = model.evaluate(x_test, y_test, batch_size=128)
        test_loss.append(le)
        test_accu.append(ae)
        print "\n The loss of the model is ",le,lr
        print "The accuracy of the model is ",ae,ar
        in_pred = model.predict_classes(x_train)
        out_pred = model.predict_classes(x_test)
        inf, inp = getPnl(in_pred, ask_train, bid_train)
        outf, outp = getPnl(out_pred, ask_test, bid_test)
        in_pnl.append(inp)
        in_flip.append(inf)
        out_pnl.append(outp)
        out_flip.append(outf)
        print "The (flip, pnl) for the in sample is ", inf, inp
        print "The (flip, pnl) for the out sample is ", outf, outp
        
    ## Plotting the results
    fig, axes = plt.subplots(2,2, figsize = (12, 12))
    fig.subplots_adjust(hspace = 0.4, wspace = 0.4)

    axes[0][0].set_title('Loss')
    axes[0][0].plot(steps, train_loss, label = 'train loss')
    axes[0][0].plot(steps, test_loss, label = 'test loss')
    axes[0][0].set_xlabel('# of steps')
    axes[0][0].legend()

    axes[0][1].set_title('Accuracy')
    axes[0][1].plot(steps, train_accu, label = 'train accuracy')
    axes[0][1].plot(steps, test_accu, label = 'test accuracy')
    axes[0][1].set_xlabel('# of steps')
    axes[0][1].legend()

    axes[1][0].set_title('# of flips')
    axes[1][0].plot(steps, in_flip, label ='in sample')
    axes[1][0].plot(steps, out_flip, label ='out sample')
    axes[1][0].set_xlabel('# of steps')
    axes[1][0].legend()

    axes[1][1].set_title('Total pnl')
    axes[1][1].plot(steps, in_pnl, label ='in sample')
    axes[1][1].plot(steps, out_pnl, label ='out sample')
    axes[1][1].set_xlabel('# of steps')
    axes[1][1].legend()
    
    plt.savefig('tx_cnn_output/convrg_rst.png')
    
    ## show model configuration
    plot_model(model, to_file = 'tx_cnn_cnn_output/model.png')
    
    


if __name__ == '__main__':
    main()
    
