from NeuralNet import NeuralNet

"""x = [[0.05, 0.05, 0.05], [0.1,  0.1,  0.1]]
b = [[[0.45], [0.45], [0.45]], [[0.8],[0.8]]]
w = [[[0.15, 0.3], [0.2, 0.35], [0.25, 0.4]], 
    [[0.5, 0.6, 0.7], [0.55, 0.65, 0.75]]]
sizes = [2,3,2]

nn = NeuralNet(sizes = sizes)
z_list, activation_list = nn.forward_propagate(x)
print("z's: ", z_list)
print("activations: ", activation_list)"""

x = [[0.05, 0.05, 0.05], [0.1,  0.1,  0.1]]
y = [[0.01, 0.01,0.01],[0.99,0.99,0.99]]
w = [[[0.15, 0.3], [0.2, 0.35], [0.25, 0.4]], 
    [[0.5, 0.6, 0.7], [0.55, 0.65, 0.75]]]
b = [[[0.45], [0.45], [0.45]], [[0.8],[0.8]]]
nn = NeuralNet(sizes = [2,3,2], biases = b, weights = w)
error, delta, dw, db, iteration = nn.train_network(x,y,0.5, minibatch_size=3, max_epoch=300)

print("error: ", error)
print("deltas: ", delta)
print("dw's: ", dw)
print("db: ", db)
print('weights: ', nn.weights)
print('biases: ', nn.biases)
print('num iterations: ', iteration)

