"""
TODO:
- doc says we can use sigmoid instaed of softmax?? ok um
- fix softmax?
- comment stuff/remove prints

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
    def backward_propagation(self, data, expected, l_rate, max_epoch = 100):
        data = np.array(data)
        z_list, activation_list = self.forward_propagate(data)
        err = activation_list[-1] - expected
        error = 0.5 * (1/data.shape[1]) * np.sum((expected-activation_list[-1])**2)
        deltas = np.ndarray((self.numLayers), dtype = object)
        db_list = np.ndarray((self.numLayers), dtype = object)
        dw_list = np.ndarray((self.numLayers), dtype = object)
        for layer in range(self.numLayers-1, 0,-1):
            #delta = np.dot(err, sigmoid_prime(z_list[layer-1]))
            delta = err * sigmoid_prime(z_list[layer-1])
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

    def batch_gradient_descent(self, data, expected, l_rate, function_type = 'sigmoid', max_epoch = 100):
        with np.errstate(divide='ignore'):
            iteration = 0
            old_error = None
            error = 0.0
            #while iteration == 0 or (iteration < max_epoch and (abs((error-old_error)/(old_error))) > 0.0001):
            while(iteration < max_epoch):
                old_error = error
                error, delta, dw, db = self.backward_propagation(data, expected, l_rate, function_type)
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
                    if(self.sizes[-1] > 1):
                        batch_expected = one_hot_encoder(train[0, i:(i+minibatch_size)])
                    else:
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
            print('expected shape: ', expected_test.shape)
            expected_test = one_hot_encoder(expected_test)    
            print('expected: ', expected_test)
            print('activation: ', activation_list[-1])       
            testing_error = 0.5 * (1/test.shape[1]) * np.sum((expected_test-activation_list[-1])**2)
            return training_error, testing_error, iteration, activation_list[-1]
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
@np.vectorize
def sigmoid(z):
    if z >=0:
        a2 = 1/(1+np.exp(-z)) 
    else:
        a2 = np.exp(z)/(1+np.exp(z))
    a2 = 1/(1+np.exp(-z)) 
    return a2

@np.vectorize
def sigmoid_prime(z):
    a = sigmoid(z)
    aprime = a*(1-a)
    return aprime

def reLU(z):
    val = np.maximum(0,z)
    return val
def reLUprime(z):
    z[z<=0] = 0
    z[z>0] = 1
    return z


def softmax(z):
    shiftz = z - np.max(z)
    e = np.exp(shiftz)
    return e / np.sum(e)

def softmax_prime(z):
    print("z shape: ", z.shape)
    Sz = softmax(z)
    print('softmax(z) shape: ', Sz.shape)
    # -SjSi can be computed using an outer product between Sz and itself. Then
    # we add back Si for the i=j cases by adding a diagonal matrix with the
    # values of Si on its diagonal.
    D = -np.outer(Sz, Sz) + np.diag(Sz.flatten())
    D = D.flatten()
    return np.reshape(D, newshape = (1, D.shape[0]))

def one_hot_encoder(expected):
    expected = expected.reshape(expected.shape[0],)
    #b = np.zeros((expected.size, int(expected.max()+1)))
    b = np.zeros((expected.size, 4))
    b[np.arange(expected.size),expected.astype(int)] = 1
    return b.T[1:,:]