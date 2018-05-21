clear
close all

HW2_Utils.getPosAndRandomNeg();
W_param = svm_main();

%create location fro saving the result
cacheFileResult = sprintf('%s/Result.mat', '../hw2data');

%generate result file for validation data
HW2_Utils.genRsltFile(W_param, 0, 'val', cacheFileResult);

%get AP and prec-rec plot on validation data
[ap, prec, rec] = HW2_Utils.cmpAP('Result.mat', 'val');