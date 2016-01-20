function neuralNet = generateNeuralNet(layers)
%generateNeuralNet generates a neural network with random weights.
%
%  *  `layers` is a vector of doubles, each number specifing the amount of
%     nodes in a layer of the network.
%
%  *  `neuralNet` is a cell array of weight matrices specifing the
%     translation from one layer of the network to the next.

  neuralNet = cell(1, length(layers)-1);

  for i = 1:length(layers)-1
    % Using random weights from -1 to 1 in order to cover the whole area of
    % the sigmoid/activation function.
    neuralNet{i} = rand(layers(i), layers(i+1)) .* 2 - 1;
  end
end
