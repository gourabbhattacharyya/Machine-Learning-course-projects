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
p = 0;      %degree of polynominal, 0 = linear
C = 10;    %margin
threshold = 0.0001;     %define sv
nSV =[];

%% Compute the Kernel
K = compute_kernel(train_data, train_data, p);

%% Calculate alpha
size_data = length(train_data(1,:));
alpha = compute_alpha(size_data, train_label, K, C);

%% Calculate b
[b, nSV] = compute_b0(train_label, alpha, K, C, threshold, nSV);

%% Calculate g(x) for training data
train_g = compute_prediction(size_data, train_label, alpha, b, K);
train_accuracy = mean(sign(train_g) == train_label);
obj_val_train = sum(sign(train_g) == sign(train_label));

%% Evaluate
%% Compute the Kernel for test set
K = compute_kernel(eval_data, train_data, p);

size_data = length(eval_data(1,:));
%% Calculate g(x) for test data
evallabel = compute_prediction(size_data, train_label, alpha, b, K);

%% Debug
if exist('eval_label', 'var') == 1
    eval_accuracy = mean(sign(evallabel) == eval_label);
    obj_val_eval = sum(sign(evallabel) == sign(eval_label));
    %disp(nSV);
    confusionMatrix = confusionmat(eval_label,evallabel);
    %disp(confusionMatrix)
    
    if C == 10
        csvwrite('confusion_matrix_10.csv',confusionMatrix);
    else
        csvwrite('confusion_matrix_01.csv',confusionMatrix);
    end
end

%% Display output message
disp('Completed.');