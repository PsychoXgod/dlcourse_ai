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
    pred = np.copy(predictions)
    if predictions.ndim == 1:
        pred -= np.max(predictions)
        probs = np.exp(pred)/np.sum(np.exp(pred))
    else:    
        pred -= np.max(predictions,axis=1).reshape(predictions.shape[0], -1)
        probs = np.exp(pred) / np.sum(np.exp(pred), axis=1, keepdims=True)
    return probs 
    # TODO implement softmax
    # Your final implementation shouldn't have any loops
    #raise Exception("Not implemented!")


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
    # TODO implement cross-entropy
    # Your final implementation shouldn't have any loops
    #raise Exception("Not implemented!")
    #target_index = target_index.ravel()
    return np.sum(-np.log(probs[np.arange(len(target_index)),target_index]))/probs.shape[0]

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
    
    target_index = target_index.ravel()
    dprediction[np.arange(len(target_index)), target_index]-=1
    dprediction/=predictions.shape[0]
    # TODO implement softmax with cross-entropy
    # Your final implementation shouldn't have any loops
    #return loss, dprediction
    #raise Exception("Not implemented!")
    

    return loss, dprediction

def l2_regularization(W, reg_strength):
    """
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    """
    # TODO: Copy from the previous assignment
    #raise Exception("Not implemented!")
    loss = reg_strength*np.sum(W*W)
    grad = 2*reg_strength*W
    return loss, grad