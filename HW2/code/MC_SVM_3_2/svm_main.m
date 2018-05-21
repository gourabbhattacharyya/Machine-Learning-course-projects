%% Task 3
%% Debug
% Check eval.mat exists before loading it
if exist('q3_1_data.mat', 'file') == 0
    error(['Error: svm_main -- File q3_1_data.mat could not be found. Please copy the ', ...
        'file into current directory and run the script again.']);
end
disp('Files found - q3_1_data.mat');

clear
close all
%% Initialize
% Load data only once
%% Load train data
load 'q3_1_data.mat';
train_data = trD;
train_label = trLb;
eval_data = valD;
eval_label = valLb;


%% Initialize defaults

C = 10;    %margin
eta0 = 1;
eta1 = 100;
maxEpoch = 2000;
train_loss_history = [];
num_classes = max(train_label(:)) + 1;
W_train = zeros(size(train_data, 1), num_classes);

%% Compute loss value for training data
for i = 1 : maxEpoch
    fprintf('Number of epoch %d: \n', i);
    learningRate = eta0/(eta1 + i);
    permutedVal = randperm(size(train_data, 2))';
    shuff_train_data = train_data(:, permutedVal);
    shuff_train_label = train_label(permutedVal, :);
    [loss_train, sumW_train, W_train] = compute_loss(shuff_train_data, shuff_train_label, W_train, learningRate, C);
    train_loss_history = [train_loss_history; loss_train];
end
disp('Loss Computed.');

[predictedLabel, train_label] = compute_prediction(train_data, W_train, train_label);
disp('Prediction Computed.');

train_accuracy = mean(predictedLabel == train_label);
train_obj_val = sum(sign(predictedLabel) == sign(train_label));




eta0 = 1;
eta1 = 100;
maxEpoch = 2000;
eval_loss_history = [];
num_classes = max(eval_label(:)) + 1;
W_eval = zeros(size(eval_data, 1), num_classes);

%% Compute loss value for validation data
for i = 1 : maxEpoch
    fprintf('Number of epoch %d: \n', i);
    learningRate = eta0/(eta1 + i);
    permutedVal = randperm(size(eval_data, 2))';
    shuff_eval_data = eval_data(:, permutedVal);
    shuff_eval_label = eval_label(permutedVal, :);
    [loss_eval, sumW_eval, W_eval] = compute_loss(shuff_eval_data, shuff_eval_label, W_eval, learningRate, C);
    eval_loss_history = [eval_loss_history; loss_eval];
end
disp('Loss Computed.');

[predictedLabelEval, eval_label] = compute_prediction(eval_data, W_eval, eval_label);
disp('Prediction Computed.');

eval_accuracy = mean(predictedLabelEval == eval_label);
eval_obj_val = sum(sign(predictedLabelEval) == sign(eval_label));
%% Display output message
disp('Completed.');


figure, plot(train_loss_history);
xlabel('#ofEpochs');
ylabel('lossForTrain');


figure, plot(eval_loss_history);
xlabel('#OfEpochs');
ylabel('lossForEval');