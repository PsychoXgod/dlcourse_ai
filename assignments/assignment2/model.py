import numpy as np

#from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization
from layers import FullyConnectedLayer, ReLULayer, l2_regularization
from loss_function import softmax_with_cross_entropy, softmax
class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        self.first_layer = FullyConnectedLayer(n_input, hidden_layer_size)
        self.second_layer = FullyConnectedLayer(hidden_layer_size, n_output)
        self.relu_layer = ReLULayer()
        # TODO Create necessary layers
        #raise Exception("Not implemented!")
    
    def _forward_pass(self, X):
        
        forward_pass_of_first_layer = self.first_layer.forward(X)
        forward_pass_of_second_layer = self.second_layer.forward(self.relu_layer.forward(forward_pass_of_first_layer))
        return forward_pass_of_first_layer, forward_pass_of_second_layer
   
    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        # TODO Set parameter gradient to zeros
        # Hint: using self.params() might be useful!
        #raise Exception("Not implemented!")
        
        self.first_layer.W.grad = np.zeros_like(self.first_layer.W.value)
        self.first_layer.B.grad = np.zeros_like(self.first_layer.B.value)
        self.second_layer.W.grad = np.zeros_like(self.second_layer.W.value)
        self.second_layer.B.grad = np.zeros_like(self.second_layer.B.value)
        
        
        forward_pass_of_first_layer, forward_pass_of_second_layer = self._forward_pass(X) 
        
        loss, dpred = softmax_with_cross_entropy(forward_pass_of_second_layer, y)
       
        backward_second_layer = self.second_layer.backward(dpred)
        backward_relu_layer = self.relu_layer.backward(backward_second_layer)
        self.first_layer.backward(backward_relu_layer)
        
        
        
        l2_loss_W1, l2_grad_W1 = l2_regularization(self.first_layer.W.value, self.reg)
        l2_loss_B1, l2_grad_B1 = l2_regularization(self.first_layer.B.value, self.reg)
        l2_loss_W2, l2_grad_W2 = l2_regularization(self.second_layer.W.value, self.reg)
        l2_loss_B2, l2_grad_B2 = l2_regularization(self.second_layer.B.value, self.reg)
        
        l2 = l2_loss_W1 + l2_loss_B1 + l2_loss_W2 + l2_loss_B2
        loss += l2
        
        self.first_layer.W.grad+= l2_grad_W1
        self.first_layer.B.grad+= l2_grad_B1
        self.second_layer.W.grad+= l2_grad_W2
        self.second_layer.B.grad+= l2_grad_B2
        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model
        
        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!
        #raise Exception("Not implemented!")
    
            
        return loss
    
        
    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # TODO: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        pred = np.zeros(X.shape[0], np.int)
        pred = np.argmax(softmax(self._forward_pass(X)[1]), axis=1)
        #raise Exception("Not implemented!")
        return pred

    def params(self):
        result = {'W_first_layer': self.first_layer.W, 'B_first_layer': self.first_layer.B,
                 'W_second_layer': self.second_layer.W, 'B_second_layer': self.second_layer.B}
        
        # TODO Implement aggregating all of the params

        #raise Exception("Not implemented!")

        return result
