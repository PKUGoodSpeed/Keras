'''
I am trying to train an RNN for word classification with in a sentence.
There are some interesting things happened here
I found that there are totally 140 classes. 
In the training sample, these classes
are imbalacely distributed. For examle, class 0 appears 28890 times, but class 1 only
appears 10 times.
Therefore, I decide to use the class_weights arguments in the model.fit() call.
First, I tried the class_weight computation function provided by sklearn, for which,
w[i] ~ 1./ count(i). However, using this class weights makes the model terrible. The accuracy
never goes above 0.1.
Then I tried a strange one: w[i] ~ 1./np.sqrt(count(i)). This sets of weights works unexpectly well.
Just after 5 epochs, the accuracy already gets close to 0.99 for both training and testing sets.
'''

import cPickle
import os
import gzip
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils, plot_model
from keras.layers.recurrent import SimpleRNN, LSTM #Actually in this test, SimpleRNN works much better
from keras.layers.embeddings import Embedding

class RNNPlayGround:
    '''
    A simple RNN model for word identification/classification
    '''
    __x_train = None
    __y_train = None
    __x_test = None
    __y_train = None
    __dicts = None
    __n_cls = None
    __cls_wts = None
    __model = None
    def __init__(self):
        '''
        Getting data from http://lisaweb.iro.umontreal.ca/transfert/lisa/users/mesnilgr/atis/
        '''
        if not os.path.isdir("./data"):
            os.system('mkdir data')
        os.system('wget http://www-etud.iro.umontreal.ca/~mesnilgr/atis/atis.fold3.pkl.gz')
        os.system('mv atis.fold3.pkl.gz ./data/')
        train, test, _, self.__dicts = cPickle.load(gzip.open('data/atis.fold3.pkl.gz','rb'))
        
        self.__x_train = train[0]
        self.__y_train = np.concatenate(train[1]) ## Here you can chose either 1 or 2
        self.__x_test = test[0]
        self.__y_test = np.concatenate(test[1])
        self.__n_cls = max(max(self.__y_train), max(self.__y_test)) + 1
        
    def contextWin(self, l, win = 5):
        '''
        Parsing the sentence into sliding windows
        '''
        assert win % 2
        assert win >= 1
        l = list(l)
        lpadded = win // 2 * [-1] + l + win // 2 * [-1]
        out = [lpadded[i:(i + win)] for i in range(len(l))]
        assert len(out) == len(l)
        return out
    
    def comp_cls_wts(self, y, n_cls):
        ''' computing class weights '''
        cls = dict((k,1.*len(y)/n_cls/np.sqrt(list(y).count(k))) for k in set(y))
        return cls
    
    def preProcess(self, win = 5):
        '''
        Creating trainning sets and testing sets
        '''
        self.__x_train = np.concatenate([self.contextWin(vec, win) for vec in self.__x_train])
        self.__x_test = np.concatenate([self.contextWin(vec, win) for vec in self.__x_test])
        self.__cls_wts = self.comp_cls_wts(self.__y_train, self.__n_cls)
        self.__y_train = np_utils.to_categorical(self.__y_train, self.__n_cls)
        self.__y_test = np_utils.to_categorical(self.__y_test, self.__n_cls)
        print self.__cls_wts
        
    def buildModel(self, win = 5):
        '''
        Build RNN model
        '''
        self.__model = Sequential()
        self.__model.add(Embedding(input_dim=1000, output_dim=1024, input_length=5))
        self.__model.add(SimpleRNN(256))
        #self.__model.add(LSTM(32))
        self.__model.add(Dense(self.__n_cls))
        self.__model.add(Activation('softmax'))
        self.__model.summary()
        
        ### Compile the model
        optimizer = SGD()
        metrics = ['accuracy']
        loss = 'categorical_crossentropy'
        self.__model.compile(optimizer = optimizer, loss = loss, metrics = metrics)
    
    def trainModel(self, epochs = 10):
        res = self.__model.fit(self.__x_train, self.__y_train, batch_size = 64,
        epochs = epochs, validation_data = (self.__x_test, self.__y_test),
        class_weight = self.__cls_wts)
        loss, accu = self.__model.evaluate(self.__x_test, self.__y_test, batch_size = 128)
        print('\n\n')
        print('The loss of this model is ', loss)
        print('The accuracy of this model is ', accu)
        
    def showModel(self):
        ## Visualize the model configuration
        plot_model(self.__model, to_file='model.png')
        
        
if __name__ == '__main__':
    rnn = RNNPlayGround()
    rnn.preProcess()
    rnn.buildModel()
    rnn.trainModel()
    rnn.showModel()
        

