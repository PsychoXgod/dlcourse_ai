def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0
    correct = 0
    true_positive = 0
    false_positive = 0
    false_negetives = 0
    true_negetives = 0
    for i in range(prediction.shape[0]):
        if prediction[i] == ground_truth[i]:
            correct+=1
        if prediction[i] == ground_truth[i] == True:
            true_positive+= 1
        if prediction[i] == True and prediction[i] != ground_truth[i]:
            false_positive+= 1
        if prediction[i] == ground_truth[i] == False:
            true_negetives+= 1
        if prediction[i] == False and  prediction[i] != ground_truth[i]:
            false_negetives+= 1   
    #print("True Positive: %f, False Negetive: %f" %(true_positive, false_negetives))        
    accuracy = correct / prediction.shape[0]
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negetives)
    f1 = 2*(precision*recall) / (precision + recall)
    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    
    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels
    
    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    correct = 0
    for i in range(prediction.shape[0]):
        if prediction[i] == ground_truth[i]:
            correct+=1
    accuracy = correct / prediction.shape[0]
    return accuracy
    # TODO: Implement computing accuracy
    return 0
