%% Initialization
clear variables
addpath('..')


%% Generate network with random weights
neuralNet = generateNeuralNet([2 2 2 1]);


%% Train network
% Use recursive multi-layer neural network implementation.

input = rand(1000, 2);
target = double([sum(input, 2) > 0.7 * 2]);

neuralNet = trainNeuralNet(neuralNet, 5000, input, target);
