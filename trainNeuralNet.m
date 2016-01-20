function neuralNet = trainNeuralNet(neuralNet, iterations, input, target)
%trainNeuralNetStatic trains a given neural network of three layers using a 
% recursive approach for an (theoretically) unlimited amount of layers.
%
%`generateNeuralNet` function trains the `neuralNet` `iterations` times 
% using the mapping from `input` to `targetOutput` cell values.
%   
% *  `neuralNet` contains a cell array length 2 of individual weight 
%    matrixes each describing the transition from one layer to the next.
% *  `iterations` is an integer greater 0.
% *  `input` is a cell array (1d) of possible input vectors (`1*n double`)
% *  `targetOutput` is a cell array of expected output vectors to those
%    in `input`.
%
% *Modeled after the non-recursive approach from 7.3.3 in R. Rojas' book
% 'Neural Networks - A Systematic Introduction'.*

  % Should be something small in order to not overshoot the goal.
  LEARNINGRATE = .01;
  
  % Total error value on which to break.
  EPS = 0.0001;
  
  
  % Placeholder. Error will be calculated in every iteration.
  totalError = Inf;

  % Train the network `iterations` times.
  display('Training...');
  for iter = 1:iterations
    sets = size(input, 1);
    totalError = totalError / sets;
    
    % Tell us about the progress every 1000 iterations.
    if (mod(iter, 1000) == 0)
      display([int2str(iter), ': ', num2str(totalError)])
    end
    
    % Stop training when the total error is good enough.
    if (totalError < EPS)
      break;
    end
    
    % Reset total error.
    totalError = 0;
  
    % Train network using each input/targetoutput pair. 
    for set = 1:sets
      targetOutput = target(set, :);

      [~, values] = applyNeuralNet(neuralNet, input(set, :));
      totalError = totalError + sum(abs((targetOutput - values{end})))/set;
      
      % Initial error on output layer.
      e = (values{end} - targetOutput)';
      
      % Backpropagate through network -- iterate over the weight layer
      % from right to left.
      for i = length(neuralNet):-1:1;
        out = values{i+1}; % Current weight layers original output.
        in = values{i}; % Current weight layers original input.
        
        delta = diag(out .* (1 - out)) * e;
        e = neuralNet{i} * delta; % Next error for backpropagation.
        neuralNet{i} = neuralNet{i} - (LEARNINGRATE * delta * in)';
      end
    end
  end
  
  % Tell us about the result.
  display(['Stopped training at ', int2str(iter), ...
    ' iterations with a total error of ', num2str(totalError)])
end

