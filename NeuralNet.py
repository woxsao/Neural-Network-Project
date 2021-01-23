import numpy as np
from numpy.core.fromnumeric import squeeze



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
        print('weights: ', self.weights)
        print('biases: ', self.biases)
    """
    This function returns the z's and activations at each layer
    Parameters:
    - data: the input data points
    Returns:
    - list of z's and activations at each layer where layer 0 is the input layer
    """
    def forward_propagate(self, data, function_type = 'sigmoid'):
        inputs = np.array(data)
        z_list = []
        activation_list = []
        activation_list.append(inputs)
        for layer in range(0, self.numLayers-1):
            z = activate(inputs, self.weights[layer], self.biases[layer])
            z_list.append(z) 
            a1 = activation_func(z, func = function_type)
            activation_list.append(a1)
            inputs = a1
        return z_list, activation_list

#tests for convergence: use max epochs and if change in delta is below a certain threshold
    def backward_propagation(self, data, expected, l_rate, minibatch_size = 1, function_type = 'sigmoid'):
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
            self.biases[layer-1]-=(l_rate/minibatch_size)  * db
            self.weights[layer-1]-= (l_rate/minibatch_size) * dw
        return error, deltas[1:], dw_list[1:], db_list[1:]
    def train_network(self, data, expected, l_rate, minibatch_size = 1, function_type = 'sigmoid', max_epoch = 100):
        iteration = 0
        old_error = None
        error = 0.0
        delta = None
        dw = None 
        db = None
        while iteration == 0 or (iteration < max_epoch and (abs((error-old_error)/(old_error))) > 0.01):
            old_error = error
            error, delta, dw, db = self.backward_propagation(data, expected, l_rate, minibatch_size, function_type)
            iteration += 1
        return error, delta, dw, db, iteration   
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
    else:
        return reLU(z)
def activation_prime(z, func = 'sigmoid'):
    if func == 'sigmoid':
        return sigmoid_prime(z)
    else:
        return relUprime(z)
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
    if(z >= 0):
        return z
    else:
        return 0

def relUprime(z):
    if z >=0:
        return 1
    else:
        return 0

