%% Neural network classifying numbers into being bigger or smaller than 0.5
% Output `[1 0]` signals 100% confidence for input being < than `0.5`.
% Output `[0 1]` signals 100% confidence for input being > than `0.5`.

%% Initialization
clear variables
addpath('..')


%% Generate network with random weights
neuralNet = generateNeuralNet([1 3 2]);


%% Train network
% Use recursive multi-layer neural network implementation.

input = rand(25, 1);
target = double([input < .5, input > .5]);

neuralNet = trainNeuralNet(neuralNet, 50000, input, target);


%% Test trained network
for r = 0:.025:1
  decision = applyNeuralNet(neuralNet, r);
  decision = decision(1) - decision(2);
  if decision > .1
    display([num2str(r), ' is smaller than 0.5 with confidence ', ...
      num2str(abs(decision))])
  elseif decision < -.1
    display([num2str(r), ' is bigger than 0.5 with confidence ', ...
      num2str(abs(decision))])
  else
    display([num2str(r), ' is.. rather equal? Confidence only ', ...
      num2str(abs(decision))])
  end
end
