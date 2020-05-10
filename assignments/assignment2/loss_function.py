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
    predictions -= np.max(predictions)
    array_exp = np.exp(predictions)
    probs = array_exp / np.sum(array_exp, axis=1, keepdims=True)
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