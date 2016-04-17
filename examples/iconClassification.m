%% Icon classification into 'die' and 'no-die' groups


%% Initialization
addpath('..')


%% Read lots and lots of icons.
d  = im2double(rgb2gray(imread('icons/dice.png')));
d1 = im2double(rgb2gray(imread('icons/dice-1.png')));
d2 = im2double(rgb2gray(imread('icons/dice-2.png')));
d3 = im2double(rgb2gray(imread('icons/dice-3.png')));
d4 = im2double(rgb2gray(imread('icons/dice-4.png')));
d5 = im2double(rgb2gray(imread('icons/dice-5.png')));
d6 = im2double(rgb2gray(imread('icons/dice-6.png')));
cat = im2double(rgb2gray(imread('icons/cat.png')));
camera = im2double(rgb2gray(imread('icons/camera.png')));
heart = im2double(rgb2gray(imread('icons/heart.png')));
plane = im2double(rgb2gray(imread('icons/airplane.png')));
alarm = im2double(rgb2gray(imread('icons/alarm.png')));
cloud = im2double(rgb2gray(imread('icons/cloud.png')));


%% Generate neural network with random weights

% All icons have the same proportions -- required condition in order
% to have an input layer of static size.
n = length(d1(:));

neuralNet = generateNeuralNet([n 200 100 2]);


%% Train neural network
neuralNet = trainNeuralNet(neuralNet, 25000, [
  d1(:)'; d2(:)'; d3(:)'; alarm(:)'; cloud(:)'; camera(:)'
], [
  [1 0]; [1 0]; [1 0]; [0 1]; [0 1]; [0 1]
]);


%% Evaluate neural network performance
% Using icons which have not been used for training.

% 4 eye die.
subplot(2,3,1)
imshow(d4)
d4r = applyNeuralNet(neuralNet, d4(:));
d4r = d4r(1) - d4r(2);
title(num2str(d4r));

% 5 eye die.
subplot(2,3,2)
imshow(d5)
d5r = applyNeuralNet(neuralNet, d5(:));
d5r = d5r(1) - d5r(2);
title(num2str(d5r));

% 6 eye die.
subplot(2,3,3)
imshow(d6)
d6r = applyNeuralNet(neuralNet, d6(:));
d6r = d6r(1) - d6r(2);
title(num2str(d6r));

% Plane.
subplot(2,3,4)
imshow(plane)
planer = applyNeuralNet(neuralNet, plane(:));
planer = planer(1) - planer(2);
title(num2str(planer));

% Cat.
subplot(2,3,5)
imshow(cat)
catr = applyNeuralNet(neuralNet, cat(:));
catr = catr(1) - catr(2);
title(num2str(catr));

% 3D die.
subplot(2,3,6)
imshow(d)
dr = applyNeuralNet(neuralNet, d(:));
dr = dr(1) - dr(2);
title(num2str(dr));
