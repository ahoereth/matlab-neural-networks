%% Simple Neural Net
% In this Script there is only one input one hidden and one output neuron
% maybe this can be generalized
clear variables;
close all;

%classify between things > 0.5 and < 0.5
trainInput = [0,1,0,1,0,1,0,0,1,1];
trainData  = [0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2];

%store the output from input,hidden and output layer
networkVals = zeros(length(trainInput),3);
networkDerivs = zeros(length(trainInput),3);

%randomly initialized values
weight1 = rand;
weight2 = rand;

learn_rate = 0.5;

total_err = inf;
iter = 0;

%Trainingphase
while total_err >= 0.001 && iter < 100000
    if(mod(iter,10000) == 0)
        total_err
    end
    total_err = 0;
    for i= 1:length(trainInput)
        %propagate input through network and safe derivativs
        networkVals(i,1) = 1/(1+exp(-trainInput(i)));
        networkDerivs(i,1) = networkVals(i,1)*(1-networkVals(i,1));
        input2 = weight1 * networkVals(i,1);

        networkVals(i,2) = 1/(1+exp(-input2));
        networkDerivs(i,2) = networkVals(i,2)*(1-networkVals(i,2));
        input3 = weight2 * networkVals(i,2);

        networkVals(i,3) = 1/(1+exp(-input3));
        networkDerivs(i,3) = networkVals(i,3)*(1-networkVals(i,3));

        %Now that we have the outputs we can calculate the error
        total_err = total_err + abs((networkVals(i,3) - trainData(i)))/length(trainData);

        % calculate errors in network
        error_D2 = networkDerivs(i,3)*(networkVals(i,3)-trainData(i));
        error_w2 = error_D2*networkVals(i,2);

        error_D1 = networkDerivs(i,2) * weight2 * error_D2;
        error_w1 = error_D1 * networkVals(i,1);

        %update weights
        weight1 = weight1 - learn_rate * error_w1;
        weight2 = weight2 - learn_rate * error_w2;

    end
    iter = iter + 1;
end

%% Testing the classification with new data
%testData = [0.3, 0.7, 0.65, 0.22, 0.93];
testData = [0, 0, 0, 1, 1];

for j = 1:length(testData)
    %propagate input through network and safe derivativs
    networkVals(j,1) = 1/(1+exp(trainInput(j)));
    input2 = weight1 * networkVals(j,1);
    networkVals(j,2) = 1/(1+exp(input2));
    input3 = weight2 * networkVals(j,2);
    networkVals(j,3) = 1/(1+exp(input3));
    disp([num2str(testData(j)), ' network says: ', num2str(networkVals(j,3))]);
end
