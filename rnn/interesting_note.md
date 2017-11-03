I am trying to train an RNN for word classification with in a sentence. 
There are some interesting things happened here.
I found that there are totally 140 classes. 
In the training sample, these classes
are imbalacely distributed. For examle, class 0 appears 28890 times, but class 1 only
appears 10 times.
Therefore, I decide to use the class_weights arguments in the model.fit() call.
1. First, I tried the class_weight computation function provided by sklearn, for which,
w[i] ~ 1./ count(i). However, using this class weights makes the model terrible. The accuracy
never goes above 0.1.
2. Then I tried a strange one: w[i] ~ 1./np.sqrt(count(i)). This sets of weights works unexpectly well.
Just after 5 epochs, the accuracy already gets close to 0.99 for both training and testing sets.
