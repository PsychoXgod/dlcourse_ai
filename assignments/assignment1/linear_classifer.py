import numpy as np


def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    '''
    '''
    pred = np.copy(predictions)
    if predictions.ndim == 1:
        pred-= np.max(predictions)
        array_of_exp = np.exp(pred)
        sum_of_exp = np.sum(array_of_exp)
        probs = array_of_exp / (np.full(predictions.shape, sum_of_exp))
    #print (probs.shape == predictions.shape)
    else:
        #pred -=np.max(pred)
        pred -= np.max(pred, axis=1).reshape(predictions.shape[0], -1)
        array_of_exp = np.exp(pred)
        sum_of_exp = np.sum(array_of_exp,axis=1)
        #print(array_of_exp.shape)
        #print(sum_of_exp.shape)
        #print(e.shape)
        probs = array_of_exp / (np.zeros(pred.shape) + sum_of_exp.reshape(sum_of_exp.shape[0],-1))
    return probs 
    # TODO implement softmax
    # Your final implementation shouldn't have any loops
    #raise Exception("Not implemented!")

    '''
    pred = np.copy(predictions)
    if predictions.ndim == 1:
        pred -= np.max(predictions)
        probs = np.exp(pred)/np.sum(np.exp(pred))
    else:    
        pred -= np.max(predictions,axis=1).reshape(predictions.shape[0], -1)
        probs = np.exp(pred) / np.sum(np.exp(pred), axis=1, keepdims=True)
    return probs 
def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''
    if probs.ndim == 1:
        return np.sum(-np.log(probs[target_index]))
    #print(target_index)
    #print(probs)
    # TODO implement cross-entropy
    # Your final implementation shouldn't have any loops
    #raise Exception("Not implemented!")
    #return np.sum(-np.log(np.take_along_axis(probs,target_index,axis=1))) 
    target_index = target_index.ravel()
    return np.sum(-np.log(probs[np.arange(len(target_index)),target_index]))/(probs.shape[0])

def softmax_with_cross_entropy(predictions, target_index):
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''             
    probs = softmax(predictions)
    loss = cross_entropy_loss(probs,target_index)
    dprediction = np.copy(probs)
    #print(dprediction)
    #print(dprediction[target_index.flatten()])
    if probs.ndim == 1:
        dprediction[target_index] = probs[target_index] - 1
    else:
        target_index = target_index.ravel()
        #print(target_index)
        #print(dprediction)
        dprediction[np.arange(len(target_index)), target_index]-=1
        dprediction/=predictions.shape[0]
    # TODO implement softmax with cross-entropy
    # Your final implementation shouldn't have any loops
    #return loss, dprediction
    #raise Exception("Not implemented!")

    return loss, dprediction


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''
    loss = reg_strength*np.sum(W * W)
    grad = 2*W*reg_strength
    #print(grad.shape)
    # TODO: implement l2 regularization and gradient
    # Your final implementation shouldn't have any loops
    #raise Exception("Not implemented!")

    return loss, grad
    

def linear_softmax(X, W, target_index):
    '''
    Performs linear classification and returns loss and gradient over W

    Arguments:
      X, np array, shape (num_batch, num_features) - batch of images
      W, np array, shape (num_features, classes) - weights
      target_index, np array, shape (num_batch) - index of target classes

    Returns:
      loss, single value - cross-entropy loss
      gradient, np.array same shape as W - gradient of weight by loss

    '''
    predictions = np.dot(X, W)
    #print(predictions)
    loss, dprediction = softmax_with_cross_entropy(predictions, target_index)
    dW = np.dot(X.T,dprediction)
    #print(probs)
    #print(X.shape, dprediction.shape)
    # TODO implement prediction and gradient over W
    # Your final implementation shouldn't have any loops
    #raise Exception("Not implemented!")
    
    return loss, dW


class LinearSoftmaxClassifier():
    def __init__(self):
        self.W = None

    def fit(self, X, y, batch_size=100, learning_rate=1e-7, reg=1e-5,
            epochs=1):
        '''
        Trains linear classifier
        
        Arguments:
          X, np array (num_samples, num_features) - training data
          y, np array of int (num_samples) - labels
          batch_size, int - batch size to use
          learning_rate, float - learning rate for gradient descent
          reg, float - L2 regularization strength
          epochs, int - number of epochs
        '''

        num_train = X.shape[0]
        num_features = X.shape[1]
        num_classes = np.max(y)+1
        if self.W is None:
            self.W = 0.001 * np.random.randn(num_features, num_classes)

        loss_history = []
        for epoch in range(epochs):
            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)
            sections = np.arange(batch_size, num_train, batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)
            
            
            epoch_X = np.array([X[i] for i in batches_indices])
            epoch_y = np.array([y[i] for i in batches_indices])
            loss_sum_batch = 0
            for index_of_batch in range(epoch_X.shape[0]):
                mini_batch_X = epoch_X[index_of_batch]
                mini_batch_y = epoch_y[index_of_batch]
                #grad_and_loss = (linear_softmax(mini_batch_X,self.W, mini_batch_y)+l2_regularization(self.W,reg)) 
                #loss_of_batch , grad_of_batch = linear_softmax(mini_batch_X,self.W, mini_batch_y) + l2_regularization(self.W,reg)
                #loss_of_batch/=2
                #loss_of_batch = grad_and_loss[0]
                #grad_of_batch = grad_and_loss[1]
                loss, dW = linear_softmax(mini_batch_X,self.W, mini_batch_y)
                _, grad_of_regulization = l2_regularization(self.W,reg)
                self.W = self.W - learning_rate*(dW + grad_of_regulization)
                loss_sum_batch += loss
                
                
            loss_history.append(loss_sum_batch/epoch_X.shape[0])
            # TODO implement generating batches from indices
            # Compute loss and gradients
            # Apply gradient to weights using learning rate
            # Don't forget to add both cross-entropy loss
            # and regularization!
            #raise Exception("Not implemented!")
            # end
            #print("Epoch %i, loss: %f" % (epoch, loss))

        return loss_history

    def predict(self, X):
        '''
        Produces classifier predictions on the set
       
        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        '''
        y_pred = np.zeros(X.shape[0], dtype=np.int)
        
        y_pred = softmax(np.dot(X, self.W)).argmax(axis=1)
        # TODO Implement class prediction
        # Your final implementation shouldn't have any loops
        #raise Exception("Not implemented!")
        
        return y_pred



                
                                                          

            

                
