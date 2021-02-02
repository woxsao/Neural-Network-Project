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



from NeuralNet import NeuralNet
import pandas as pd 
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
import seaborn as sns 
import matplotlib.pyplot as plt

def one_hot_encoder(expected):
    expected = expected.reshape(expected.shape[0],)
    #b = np.zeros((expected.size, int(expected.max()+1)))
    b = np.zeros((expected.size, 4))
    b[np.arange(expected.size),expected.astype(int)] = 1
    return b.T[1:,:]

#@np.vectorize
def one_hot_decoder(activation):
    activation_zeros = np.zeros(activation.shape)
    activation_t = activation.T
    for i in range(0, activation_t.shape[0]):
        maximum_index = np.argmax(activation_t[i])
        activation_zeros[maximum_index,i] = 1
    return activation_zeros

def translate_into_classes(activation):
    classes = []
    for i in range(0,activation.shape[1]):
        maximum_index = np.argmax(activation[:, i])
        classes.append(maximum_index+1)
    return classes

scaler = preprocessing.MinMaxScaler()
wine_data_csv = pd.read_csv('wine.data.csv')
predicted = wine_data_csv[wine_data_csv.columns[0]]
wine_data = wine_data_csv
wine_data[wine_data.columns[1:]] = scaler.fit_transform(wine_data[wine_data.columns[1:]])
#wine_data = (wine_data-wine_data.mean())/wine_data.std()
wine_data.iloc[:,0] = predicted
sample_size = 130

#shuffling data along rows, then transpose so features are rows and samples are columns
wine_data=(wine_data.sample(frac=1).reset_index(drop=True)).T

#split train and test 
wine_data_train = wine_data.iloc[:,:sample_size].to_numpy()

wine_data_test = wine_data.iloc[:,sample_size:].to_numpy()

wine_data.to_csv("training_data.csv", index = False)

print('shape of train: ', wine_data_train.shape)
#BATCH GRADIENT DESCENT
#Start with 1 hidden layer, 1 output node and average number of nodes between input and output 
hidden_layer_size = int((2.0/3)* wine_data_train.shape[0] + 3)
max_epoch = 8000
categories = [1,2,3]

nn2 = NeuralNet(sizes = [13, hidden_layer_size,8, 3])
training_error, iteration = nn2.descent_with_momentum(wine_data_train[1:,:], wine_data_train[0,:], l_rate = .2, momentum_val = 0.9, function_type='sigmoid', max_epoch=10000, minibatch_size=20, threshold=0.00006)
print('iteration: ', iteration-1)
z_list, activation_list = nn2.forward_propagate(wine_data_test[1:,:])
print('activation list: ', activation_list[-1][0])
activation = one_hot_decoder(activation_list[-1])
test_classified = translate_into_classes(activation_list[-1])
test_actual = list(map(int,wine_data_test[0,:]))
print("test actual: ", test_actual)
print("test_classified: ", test_classified)
best_confusion = confusion_matrix(test_actual, test_classified)
best_confDF = pd.DataFrame(best_confusion)
best_confDF.index = categories
best_confDF.columns = categories
plt.figure()
heatmap_best = sns.heatmap(best_confDF, annot = True)
plt.title('Confusion Matrix for Wine, 2 hidden layers')
plt.show()


expected_test = one_hot_encoder(wine_data_test[0,:])
testing_error = 0.5*(1/wine_data_test.shape[1] * np.sum((expected_test-activation)**2))


#print('activation: ', activation_list[-1])
#print('expected: ', expected_test)
print('training error: ', training_error[-1])
print("testing error: ", testing_error)
#1 hidden layer, 3 output nodes
# expected shape should be 3, n, using relU for hidden layers and softmax for output layer

plt.figure()
x = range(0,iteration)
plt.plot(x,training_error, 'b-')
plt.xlabel("Epoch #")
plt.ylabel("Training error")
plt.title("Training Error Vs. Epoch")
plt.show()





