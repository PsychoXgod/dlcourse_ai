def multiclass_accuracy(prediction, ground_truth):
    """
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    """
    #print(prediction, ground_truth)
    correct = 0
    for i in range(prediction.shape[0]):
        if prediction[i] == ground_truth[i]:
            correct+=1
    #print(correct)        
    accuracy = correct / prediction.shape[0]
    return accuracy
    # TODO: Implement computing accuracy
    raise Exception("Not implemented!")

    return 0
