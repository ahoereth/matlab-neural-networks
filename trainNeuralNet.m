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
% *  If `targetOutput` is omitted, `input` is expected to be a cell array 
%    of individual input -> targetOutput mappings contained in their own 
%    cell arrays. Example: `input = { {[0, 1], [1, 0]}, ... }`, where the 
%    network is to be trained to map `[0, 1]` to `[1, 0]`.
%
% *Modeled after the non-recursive approach from 7.3.3 in R. Rojas' book
% 'Neural Networks - A Systematic Introduction'.*

  % Should be something small in order to not overshoot the goal.
  LEARNINGRATE = .01;
  
  % Total error value on which to break.
  EPS = .01;
  
  % Translate single input/target cell array argument to individual 
  % input and target cell arrays.
  if nargin < 4
    inputoutput = input;
    input = cell(1, length(inputoutput));
    target = cell(1, length(inputoutput));
    for i = 1:length(inputoutput)
      input{i} = inputoutput{i}{1};
      target{i} = inputoutput{i}{2};
    end
  end
  
  % Placeholder. Error will be calculated in every iteration.
  totalError = Inf;

  % Train the network `iterations` times.
  display('Training...');
  for iter = 1:iterations
    
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
    for sample = 1:length(input)
      [~, values] = applyNeuralNet(neuralNet, input{sample});
      totalError = totalError + sum((target{sample} - values{end}).^2 / 2);
      
      % Initial error on output layer.
      e = (values{end} - target{sample})';
      
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

