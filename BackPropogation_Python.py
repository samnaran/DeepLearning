## Back Propogation Using Python

import numpy as np

# Define a simple neural network with a single layer
class SimpleNeuralNetwork:
    def __init__(self, num_inputs, num_neurons):
        # Initialize the weights and biases randomly
        self.weights = np.random.rand(num_inputs, num_neurons)
        self.biases = np.random.rand(num_neurons)

    # Define the forward pass
    def forward(self, inputs):
        # Compute the dot product of inputs and weights plus biases
        return np.dot(inputs, self.weights) + self.biases

    # Define the backward pass (backpropagation)
    def backward(self, inputs, targets, learning_rate):
        # Compute the deltas for the weights and biases
        output = self.forward(inputs)
        delta = targets - output
        weight_deltas = np.dot(inputs.T, delta)
        bias_delta = delta
        # Update the weights and biases
        self.weights -= learning_rate * weight_deltas
        # Calculate the mean of the bias deltas along the first axis (samples)
        self.biases -= learning_rate * np.mean(bias_delta, axis=0) 

# Define the training data
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([[0], [1], [1], [0]])

# Create an instance of the neural network
nn = SimpleNeuralNetwork(2, 1)

# Define the number of training iterations
num_iterations = 10000

# Define the learning rate
learning_rate = 0.1

# Train the network
for i in range(num_iterations):
    nn.backward(inputs, targets, learning_rate)
    if i % 1000 == 0:
        print(f"Iteration {i}: Weights = {nn.weights}, Biases = {nn.biases}")
