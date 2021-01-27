"""
TODO:
- put in the graphs
- do the right import things
- comment stuff/remove prints
- try more configurations for different hidden layer numbers
- import feature names
-import classification names
- update the train_network to return activations at the last layer for test
- update train)network to return the predicted of the test group 
- rearrange activation at last layer to be of shape (3,)
"""

from random import sample
from NeuralNet import NeuralNet
import pandas as pd 
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA as sklearnPCA


#normalizing data
scaler = preprocessing.MinMaxScaler()
wine_data_csv = pd.read_csv('wine.data.csv')
predicted = wine_data_csv[wine_data_csv.columns[0]]
wine_data = wine_data_csv
wine_data[wine_data.columns[1:]] = scaler.fit_transform(wine_data[wine_data.columns[1:]])
#wine_data = (wine_data-wine_data.mean())/wine_data.std()
wine_data.iloc[:,0] = predicted
sample_size = 100

#shuffling data along rows, then transpose so features are rows and samples are columns
wine_data=(wine_data.sample(frac=1).reset_index(drop=True)).T

#split train and test 
wine_data_train = wine_data.iloc[:,:sample_size].to_numpy()

wine_data_test = wine_data.iloc[:,sample_size:].to_numpy()

wine_data.to_csv("training_data.csv", index = False)

print('shape of train: ', wine_data_train.shape)
#BATCH GRADIENT DESCENT
#Start with 1 hidden layer, 1 output node and average number of nodes between input and output 
hidden_layer_size = 10
"""nn = NeuralNet(sizes = [13, hidden_layer_size, 3])

training_error,testing_error, iteration, activation_list = nn.train_network(wine_data_train, wine_data_test,l_rate = 0.001, minibatch_size=sample_size, function_type='sigmoid', max_epoch = 10000)

print('batch testing error: ', testing_error)
print('batch training error: ', training_error)
print('batch iterations: ', iteration)"""

#1 hidden layer, 3 output nodes
# expected shape should be 3, n, using relU for hidden layers and softmax for output layer

"""#MINIBATCH GRADIENT DESCENT
nn2 = NeuralNet(sizes = [13, hidden_layer_size, 1])
training_error, testing_error, iteration, activation_list = nn2.train_network(wine_data_train, wine_data_test, l_rate = 0.0001, minibatch_size = 20, function_type = 'sigmoid', max_epoch = 10000)
print('mini testing error: ', testing_error)
print('mini training error: ', training_error)
print('mini iterations: ', iteration)
"""
#STOCHASTIC GRADIENT DESCENT
nn3 = NeuralNet(sizes = [13, hidden_layer_size, 3])
training_error, testing_error, iteration, activation_list = nn3.train_network(wine_data_train, wine_data_test, l_rate = 0.0001, minibatch_size = 1, function_type = 'sigmoid', max_epoch = 10000)
print('stochastic testing error: ', testing_error)
print('stochastic training error: ', training_error)
print('stochastic iterations: ', iteration)




"""confusion = confusion_matrix(y, predicted)
confDF = pd.DataFrame(data = confusion)
confDF.columns = categories
confDF.index = categories
heatmap = sns.heatmap(confDF, annot = True)
plt.title("Confusion Matrix for 10-Fold Cross Validation")
plt.show()"""


"""#New Structure: 13 neuron input, 1 hidden layer, 3 neuron output
#neurons = 2/3 input layer + output layer                 
#hidden_layer_size = int(((2/3)*13)+ 3)
hidden_layer_size = (int)((13.0+1.0)/2)
nn4 = NeuralNet([13,hidden_layer_size,3])
wine_data = wine_data_csv
sample_size = 142
wine_data[wine_data.columns[1:]] = scaler.fit_transform(wine_data[wine_data.columns[1:]])
#shuffling data along rows, then transpose so features are rows and samples are columns
wine_data=(wine_data.sample(frac=1).reset_index(drop=True)).T

#split train and test 
wine_data_train = wine_data.iloc[:,:sample_size].to_numpy()

wine_data_test = wine_data.iloc[:,sample_size:].to_numpy()
training_error, testing_error, iteration = nn4.train_network(wine_data_train, wine_data_test, l_rate = 0.01, minibatch_size = 1, function_type = 'reLU', max_epoch = 1000)
print('stochastic testing error: ', testing_error)
print('stochastic training error: ', training_error)
print('stochastic iterations: ', iteration)"""




