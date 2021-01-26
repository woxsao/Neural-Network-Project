from NeuralNet import NeuralNet
import pandas as pd 
import numpy as np
from sklearn import preprocessing

#normalizing data
scaler = preprocessing.MinMaxScaler()
wine_data = pd.read_csv('wine.data.csv')
wine_data[wine_data.columns] = scaler.fit_transform(wine_data[wine_data.columns])

sample_size = 160

#shuffling data along rows, then transpose so features are rows and samples are columns
wine_data=(wine_data.sample(frac=1).reset_index(drop=True)).T

#split train and test 
wine_data_train = wine_data.iloc[:,:sample_size].to_numpy()

wine_data_test = wine_data.iloc[:,sample_size:].to_numpy()

print('shape of train: ', wine_data_train.shape)
"""BATCH GRADIENT DESCENT"""
#Start with 1 hidden layer, 1 output node and average number of nodes between input and output 
hidden_layer_size = (int)((13.0+1.0)/2)
nn = NeuralNet(sizes = [13, hidden_layer_size, 1])

training_error,testing_error, iteration = nn.train_network(wine_data_train, wine_data_test,l_rate = 0.01, minibatch_size=sample_size, function_type='reLU', max_epoch = 300)

print('batch testing error: ', testing_error)
print('batch training error: ', training_error)
print('batch iterations: ', iteration)
#1 hidden layer, 3 output nodes
# expected shape should be 3, n, using relU for hidden layers and softmax for output layer

"""MINIBATCH GRADIENT DESCENT"""
nn2 = NeuralNet(sizes = [13, hidden_layer_size, 1])
training_error, testing_error, iteration = nn2.train_network(wine_data_train, wine_data_test, l_rate = 0.01, minibatch_size = 32, function_type = 'reLU', max_epoch = 300)
print('mini testing error: ', testing_error)
print('mini training error: ', training_error)
print('mini iterations: ', iteration)

"""STOCHASTIC GRADIENT DESCENT"""
nn3 = NeuralNet(sizes = [13, hidden_layer_size, 1])
training_error, testing_error, iteration = nn3.train_network(wine_data_train, wine_data_test, l_rate = 0.01, minibatch_size = 1, function_type = 'reLU', max_epoch = 300)
print('stochastic testing error: ', testing_error)
print('stochastic training error: ', training_error)
print('stochastic iterations: ', iteration)

def one_hot_encoder(expected):
    expected = expected.to_numpy()
    expected = expected.reshape(expected.shape[0],)
    b = np.zeros((expected.size, int(expected.max()+1)))
    b[np.arange(expected.size),expected] = 1
    return b.T
"""expected = wine_data_sample.iloc[0].astype(int)
expected = one_hot_encoder(expected)[1:,:]
print(expected)
#neurons = 2/3 input layer + output layer                 
hidden_layer_size = int(((2/3)*13)+ 3)
nn = NeuralNet([13,hidden_layer_size,3])
error, delta, dw, db, iteration = nn.train_network(wine_data, expected, l_rate = 0.1, minibatch_size=34, max_epoch = 300, function_type='reLU', softmax= True)
print("final error: ", error)
print('num iterations: ', iteration)"""


