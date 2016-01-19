function [output, values] = applyNeuralNet(neuralNet, input)  
  values = cell(1, length(neuralNet)+1);

  % Save input values into first cell.
  values{1} = input(:)';

  % Apply neural network to input layer by layer.
  for i = 1:length(neuralNet)
    % Application of logistic function.
    x = values{i} * neuralNet{i};
    values{i+1} = 1./(1+exp(-(x)));
  end

  % Return output layer values individually.
  output = values{end};
end
