import numpy as np


class NeuralNet:
    def __init__(self, sizes, biases = None, weights = None):
        self.numLayers = len(sizes)
        self.sizes = sizes
        if biases == None and weights == None:
            self.weights = []
            self.biases = []
            for i in range(1, self.numLayers):
                self.weights.append(np.random.rand(sizes[i],sizes[i-1])-0.5)
                self.biases.append(np.random.rand(sizes[i],1)-0.5)
            self.weights = np.array(self.weights)
            self.biases = np.array(self.biases)
            
        else:
            self.biases = biases
            self.weights = weights
        
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

    """
    This function runs the backpropagation algorithm. 
    Parameters: 
    - data: the input data points
    - expected: the list of expected classes
    - l_rate: learning rate (float between 0-1)
    - function_type: string value indicating which activation function to use (sigmoid or reLU is what I have)
    - dw_prev: If intending on using momentum, must specify the dw's of the previous iteration
    - mometnum: float between (0-1) to affect the weight increments. 
    Returns:
    - error: training error
    - deltas at each layer
    -dw's at each layer
    -db's at each layer
    """
    def backward_propagation(self, data, expected, l_rate, function_type = 'sigmoid', dw_prev = None, momentum = 0):
        data = np.array(data)
        z_list, activation_list = self.forward_propagate(data, function_type= function_type)
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
            dw = np.dot(delta, activation_list[layer-1].T) * (l_rate/data.shape[1])
            dw_list[layer] = dw
            err = np.dot(np.array(self.weights[layer-1]).T, delta)
            self.biases[layer-1]-=(l_rate/data.shape[1])  * db
            if(dw_prev != None):
                self.weights[layer-1]-= dw + (momentum* dw_prev[layer-1])
            else:
                self.weights[layer-1]-= dw
        return error, deltas[1:], dw_list[1:], db_list[1:]

    """
    Runs minibatch gradient descent
    Parameters:
    - data: data points 
    - expected: list of classes for each point
    -l_rate: learning rate (float between 0-1)
    - momentum_val: float between 0-1 that specifies a momentum value
    -function type: string indicating which activation function to use
    - max_epoch: maximum iterations for the gradient descent
    - minibatch_size: batch size
    - threshold: error threshold percentage (0-1), the smaller the value the harder it is to converge. 
    """    
    def descent_with_momentum(self, data, expected, l_rate, momentum_val = 0,function_type = 'sigmoid', max_epoch = 100, minibatch_size = 1, threshold = 0.0001):
        with np.errstate(divide='ignore'):
            expected = one_hot_encoder(expected)
            iteration = 0
            old_error = None
            error = 0.0
            dw = None
            error_list = []
            data_full = np.append(expected,data, axis = 0)
            while iteration <2 or (iteration < max_epoch and abs((old_error-error)/(old_error)) > threshold):
                np.random.shuffle(data_full.T) 
                for i in range(0,data.shape[1], minibatch_size):
                    batch_data = data_full[3:, i:(i + minibatch_size)]
                    batch_expected = data_full[0:3, i:(i+minibatch_size)]
                    if(i == 0):
                        dw_prev = None
                    else:
                        dw_prev = dw
                    error, delta, dw, db = self.backward_propagation(batch_data, batch_expected, l_rate, function_type, dw_prev = dw_prev, momentum = momentum_val)
                    
                iteration += 1
                z_list, activation_list = self.forward_propagate(data, function_type)
                old_error = error
                error = 0.5*(1/data.shape[1] * np.sum((expected-activation_list[-1])**2))
                error_list.append(error)
                print('error: ', error, ' iteration: ', iteration)
            print('error list shape: ', error_list)
            return error_list, iteration

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

"""
This function returns the activations at a layer given a z and function type.
Parameters:
- z: the z for the layer
- func: the function type, default is sigmoid, reLU also supported.
Returns: 
- activations at the layer
"""
def activation_func(z, func = 'sigmoid'):
    if func == 'sigmoid':
        return sigmoid(z)
    else:
        return reLU(z)
"""
This function returns the derivative of the activation function to calculate delta for backprop.
Parameters:
-z: Z's at specified layer
-func: function type, default is sigmoid but reLU also supported.
Returns:
- the derivative function applied to the z's. 
"""    
def activation_prime(z, func = 'sigmoid'):
    if func == 'sigmoid':
        return sigmoid_prime(z)
    else:
        return reLUprime(z)
    
"""
This function returns  the activations at a layer using sigmoid
Parameters: 
- z: The z's at this layer 
Returns;
- the list of activations at this layer. 
"""
@np.vectorize
def sigmoid(z):
    if z >=0:
        a2 = 1/(1+np.exp(-z)) 
    else:
        a2 = np.exp(z)/(1+np.exp(z))
    a2 = 1/(1+np.exp(-z)) 
    return a2

"""
Returns the derivative function applied to z's at a layer for sigmoid.
Parameters: 
- z: The z's at this layer 
Returns;
- list of Sigmoid derivative applied to z's
"""
@np.vectorize
def sigmoid_prime(z):
    a = sigmoid(z)
    aprime = a*(1-a)
    return aprime

"""
This function returns  the activations at a layer using reLU
Parameters: 
- z: The z's at this layer 
Returns;
- the list of activations at this layer. 
"""
@np.vectorize
def reLU(z):
    if(z<0):
        return 0
    else:
        return z

"""
Returns the derivative function applied to z's at a layer for reLU.
Parameters: 
- z: The z's at this layer 
Returns;
- list of reLU derivative applied to z's
"""
@np.vectorize
def reLUprime(z):
    if(z<0):
        return 0
    else:
        return 1


"""
This function takes in an array with the expected classes of each wine and returns a 3xn array where the values
are all 0 except for the row where the wine class corresponds. For instance, if the wine sample's class is 2, the corresponding 
column would look like [0,1,0]

Parameters: 
-expected: list of n samples and their wine classes (integers running from 1....n)
Returns: 
- array of (x by n) where x is the number of classes and n is the number of samples
"""
def one_hot_encoder(expected):
    expected = expected.reshape(expected.shape[0],)
    #b = np.zeros((expected.size, int(expected.max()+1)))
    b = np.zeros((expected.size, 4))
    b[np.arange(expected.size),expected.astype(int)] = 1
    return b.T[1:,:]
