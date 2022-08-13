%% Start program
%%
%% Clear and close all
clear all;
close all;
clc;
%%
%% load and plot original data
dataSN = readmatrix('Time-domain_Rabi-antenna.csv');
t = dataSN(:,1);
intensity = dataSN(:,2);
figure(1)
% plot(t, intensity)
plot(dataSN(:,2))
xlabel("Time (fs)")
ylabel("Amplitude (a.u.)")
title("Time-domain of Rabi Antenna")
%%
%% Partition the training and test data. Train on the first 90% of the sequence and test on the last 10%.
numTimeStepsTrain = floor(0.9*numel(dataSN));
dataTrain = dataSN(1:numTimeStepsTrain+1);
dataTest = dataSN(numTimeStepsTrain+1:end);
%%
%% Standardize Data
mu = mean(dataTrain);
sig = std(dataTrain);
dataTrainStandardized = (dataTrain - mu) / sig;
%%
%% Prepare Predictors and Responses
XTrain = dataTrainStandardized(1:end-1);
YTrain = dataTrainStandardized(2:end);
%%
%% Define LSTM Network Architecture
numFeatures = 1;
numResponses = 1;
numHiddenUnits = 200;

layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer];

options = trainingOptions('adam', ...
    'MaxEpochs',250, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.001, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',125, ...
    'LearnRateDropFactor',0.2, ...
    'Verbose',0, ...
    'Plots','training-progress');
%%
%% Train LSTM Network
net = trainNetwork(XTrain,YTrain,layers,options);
%%
%% Forecast Future Time Steps
dataTestStandardized = (dataTest - mu) / sig;
XTest = dataTestStandardized(1:end-1);

net = predictAndUpdateState(net,XTrain);
[net,YPred] = predictAndUpdateState(net,YTrain(end));

numTimeStepsTest = numel(XTest);
for i = 2:numTimeStepsTest
    [net,YPred(:,i)] = predictAndUpdateState(net,YPred(:,i-1),'ExecutionEnvironment','gpu');
end

YPred = sig*YPred + mu;

YTest = dataTest(2:end);
rmse = sqrt(mean((YPred-YTest).^2))

% Plot the training time series with the forecasted values.
figure(2)
plot(dataTrain(1:end-1))
hold on
idx = numTimeStepsTrain:(numTimeStepsTrain+numTimeStepsTest);
plot(idx,[dataSN(numTimeStepsTrain) YPred],'.-')
hold off
xlabel("Time (fs)")
ylabel("Amplitude (a.u.)")
title("Forecast")
legend(["Observed" "Forecast"])

% Compare the forecasted values with the test data.
figure(3)
subplot(2,1,1)
plot(YTest)
hold on
plot(YPred,'.-')
hold off
legend(["Observed" "Forecast"])
ylabel("Amplitude (a.u.)")
title("Forecast")

subplot(2,1,2)
stem(YPred - YTest)
xlabel("Time (fs)")
ylabel("Error")
title("RMSE = " + rmse)
%%
%% Update Network State with Observed Values
net = resetState(net);
net = predictAndUpdateState(net,XTrain);

YPred = [];
numTimeStepsTest = numel(XTest);
for i = 1:numTimeStepsTest
    [net,YPred(:,i)] = predictAndUpdateState(net,XTest(:,i),'ExecutionEnvironment','gpu');
end

YPred = sig*YPred + mu;

rmse = sqrt(mean((YPred-YTest).^2))

% Plot the training time series with the forecasted values with updates LSTM network.
figure(4)
plot(dataTrain(1:end-1))
hold on
idx = numTimeStepsTrain:(numTimeStepsTrain+numTimeStepsTest);
plot(idx,[dataSN(numTimeStepsTrain) YPred],'.-')
hold off
xlabel("Time (fs)")
ylabel("Amplitude (a.u.)")
title("Forecast with Updates")
legend(["Observed" "Forecast with Updates"])

% Compare the forecasted values with the test data with updates LSTM network.
figure(5)
subplot(2,1,1)
plot(YTest)
hold on
plot(YPred,'.-')
hold off
legend(["Observed" "Predicted"])
ylabel("Amplitude (a.u.)")
title("Forecast with Updates")

subplot(2,1,2)
stem(YPred - YTest)
xlabel("Time (fs)")
ylabel("Error")
title("RMSE = " + rmse)
%%
% Error Bars Plot
figure(6)
err = YTest - YPred;
errorbar(XTest,YTest,err,'-s','MarkerSize',5,...
    'MarkerEdgeColor','red','MarkerFaceColor','red')
%% End program
