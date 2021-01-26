"""
TODO:
- Code in softmax prime
- make sURE TO CHANGE IT IN THE BACKPROP TOO
- doc says we can use sigmoid instaed of softmax?? ok um
- put in training error for minibatch
- code in testing error

"""
import numpy as np
from sklearn.utils import shuffle
import random

class NeuralNet:
    def __init__(self, sizes, biases = None, weights = None):
        self.numLayers = len(sizes)
        self.sizes = sizes
        if biases == None and weights == None:
            self.weights = []
            self.biases = []
            for i in range(1, self.numLayers):
                self.weights.append(np.random.rand(sizes[i],sizes[i-1]))
                self.biases.append(np.random.rand(sizes[i],1))
            self.weights = np.array(self.weights)
            self.biases = np.array(self.biases)
        else:
            self.biases = biases
            self.weights = weights
        #for sizes [2,3,2] I'd expect for the weights matrix to be []
        
    """
    This function returns the z's and activations at each layer
    Parameters:
    - data: the input data points
    Returns:
    - list of z's and activations at each layer where layer 0 is the input layer
    """
    def forward_propagate(self, data, function_type = 'sigmoid', softmax = False):
        inputs = np.array(data)
        z_list = []
        activation_list = []
        activation_list.append(inputs)
        for layer in range(0, self.numLayers-1):
            z = activate(inputs, self.weights[layer], self.biases[layer])
            z_list.append(z) 
            if layer == self.numLayers-2 and softmax == True:
                a1 = activation_func(z, func = 'softmax')
            else:
                a1 = activation_func(z, func = function_type)
            activation_list.append(a1)
            inputs = a1
        return z_list, activation_list

#tests for convergence: use max epochs and if change in delta is below a certain threshold
    def backward_propagation(self, data, expected, l_rate, function_type = 'sigmoid'):
        data = np.array(data)
        z_list, activation_list = self.forward_propagate(data)
        err = activation_list[-1] - expected
        error = 0.5 * (1/data.shape[1]) * np.sum((expected-activation_list[-1])**2)
        deltas = np.ndarray((self.numLayers), dtype = object)
        db_list = np.ndarray((self.numLayers), dtype = object)
        dw_list = np.ndarray((self.numLayers), dtype = object)
        for layer in range(self.numLayers-1, 0,-1):
            delta = err * activation_prime(z_list[layer-1], func = function_type)
            delta_sum = np.sum(delta, axis = 1)
            deltas[layer] = delta
            db = delta_sum
            db = db.reshape(db.shape[0],1)
            db_list[layer] = db
            dw = np.dot(delta, activation_list[layer-1].T)
            dw_list[layer] = dw
            err = np.dot(np.array(self.weights[layer-1]).T, delta)
            self.biases[layer-1]-=(l_rate/data.shape[1])  * db
            self.weights[layer-1]-= (l_rate/data.shape[1]) * dw
        return error, deltas[1:], dw_list[1:], db_list[1:]
    
    def batch_gradient_descent(self, data, l_rate, function_type = 'sigmoid', max_epoch = 100):
        with np.errstate(divide='ignore'):
            iteration = 0
            old_error = None
            error = 0.0
            delta = None
            dw = None 
            db = None
            expected = data[0, :]
            train = data[1:,:]
            while iteration == 0 or (iteration < max_epoch and (abs((error-old_error)/(old_error))) > 0.0001):
            #while(iteration < max_epoch):
                old_error = error
                error, delta, dw, db = self.backward_propagation(train, expected, l_rate, function_type)
                iteration += 1
                print('error: ', error)
            return error, iteration   

    def train_network(self, train, test, l_rate, minibatch_size = 1, function_type = 'sigmoid', max_epoch = 100):
          with np.errstate(divide='ignore'):
            iteration = 1
            old_error = 0.0
            error = 0.0
            while iteration == 1 or (iteration <= max_epoch and abs((error-old_error)/(old_error)) > 0.001):
                old_error = error
                error_list = []
                #multidimensional arrays only shuffled along first axis
                np.random.shuffle(train.T)
                for i in range(0,train.shape[1],minibatch_size):
                    batch_data = train[1:, i:(i + minibatch_size)]
                    batch_expected = np.reshape(train[0, i:(i + minibatch_size)], newshape= (1,batch_data.shape[1]))
                    error, delta, dw, db = self.backward_propagation(batch_data, batch_expected, l_rate, function_type)
                    error_list= np.append(error_list, error)
                #i think the trarining error should be the average of the last batch for all epochs?
                error = np.average(error_list)
                iteration += 1
            training_error = error
            #Testing section:
            activation_list = self.forward_propagate(test[1:,:])[1] 
            expected_test = test[0, :]           
            testing_error = 0.5 * (1/test.shape[1]) * np.sum((expected_test-activation_list[-1])**2)
            return training_error, testing_error, iteration
"""
This function returns the z's in one layer given the data, the weights, and the biases for the layer.
Parameters:
- a: an array of data points of n * m dimensions
- w: an array of weights that munst be o * n dimensions 
- b: an array of biases at the layer
Returns:
- the list of z's at this particular layer
"""
def activate(a, w, b):
    z = np.add(np.dot(w,a), b)
    return z

def activation_func(z, func = 'sigmoid'):
    if func == 'sigmoid':
        return sigmoid(z)
    elif func == 'reLU':
        return reLU(z)
    else:
        return softmax(z)
def activation_prime(z, func = 'sigmoid'):
    if func == 'sigmoid':
        return sigmoid_prime(z)
    elif func == 'reLU':
        return reLUprime(z)
    else:
        return softmax_prime(z)
"""
This function returns  the activations at a layer
Parameters: 
- z: The z's at this layer 
Returns;
- the list of activations at this layer. 
"""
def sigmoid(z):
    if z.all() >=0:
        a2 = 1/(1+np.exp(-z)) 
    else:
        a2 = np.exp(z)/(1+np.exp(z))
    return a2

def sigmoid_prime(z):
    a = sigmoid(z)
    aprime = a*(1-a)
    return aprime

def reLU(z):
    if(z.all() >= 0):
        return z
    else:
        return 0

def reLUprime(z):
    if z.all() >=0:
        return 1
    else:
        return 0

def softmax(z):
    return np.exp(z)/np.sum(np.exp(z), axis = 0)

def softmax_prime(z):
    # Reshape the 1-d softmax to 2-d so that np.dot will do the matrix multiplication
    s = z.reshape(-1,1)
    return np.diagflat(s) - np.dot(s, s.T)