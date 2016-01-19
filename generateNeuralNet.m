function neuralNet = generateNeuralNet(layers)
  neuralNet = cell(1, length(layers)-1);

  for i = 1:length(layers)-1
    neuralNet{i} = rand(layers(i), layers(i+1)) - .5;
  end
end
