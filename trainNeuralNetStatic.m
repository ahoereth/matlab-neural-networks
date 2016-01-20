function neuralNet = trainNeuralNetStatic(neuralNet, iterations, input, target)  
%trainNeuralNetStatic trains a given neural network of three layers (input,
%hidden, output) using a non recursive approach.
%
%`generateNeuralNet` function trains the `neuralNet` `iterations` times 
% using the mapping from `input` to `targetOutput` cell values.
%   
% *  `neuralNet` contains a cell array length 2 of individual weight 
%    matrixes each describing the transition from one layer to the next.
% *  `iterations` is an integer greater 0.
% *  `input` is a cell array (1d) of possible input vectors (`1*n double`)
% *  `target` is a cell array of expected output vectors to those in 
%    `input`.
%
% Very verbose code in order to closly match the referenced approach and
% variables.
%
% *Modeled after the non-recursive approach from 7.3.3 in R. Rojas' book
% 'Neural Networks - A Systematic Introduction'.*

  % Should be something small in order to not overshoot the goal.
  LEARNINGRATE = .01;
  
  % Total error value on which to break.
  EPS = .01;
  
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
  
    sets = size(input, 1);

    % Train network using each input/targetoutput pair. 
    for set = 1:sets
      t = target(set, :); % target output regarding current input

      [~, values] = applyNeuralNet(neuralNet, input(set, :));
      totalError = totalError + sum(abs((t - values{end})))/sets;

      o2 = values{3}; % output values
      o1 = values{2}; % hidden layer values
      o  = values{1}; % input values
      D2 = diag(o2 .* (1 - o2)); % matrix of derivatives for output layer
      D1 = diag(o1 .* (1 - o1)); % matrix of derivatives for hidden layer
      W2 = neuralNet{2}; % weights from hidden to output
      W1 = neuralNet{1}; % weights from input to hidden
      e = o2(:) - t(:); % deviation
      delta2 = D2 * e;
      delta1 = D1 * W2 * delta2;
      
      % Correction matrices.
      correctionsW2 = -LEARNINGRATE * delta2 * o1;
      correctionsW1 = -LEARNINGRATE * delta1 * o;
      
      % Weight updates.
      neuralNet{2} = W2 + correctionsW2';
      neuralNet{1} = W1 + correctionsW1';
    end
  end
  
  % Tell us about the result.
  display(['Stopped training at ', int2str(iter), ...
    ' iterations with a total error of ', num2str(totalError)])
end

