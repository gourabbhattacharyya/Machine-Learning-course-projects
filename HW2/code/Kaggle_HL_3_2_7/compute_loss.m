function [ loss, sumW, W ] = compute_loss(data, label, W, learningRate, C)

%compute loss
    %% Initialize inputs
    
    num_train = size(data, 2);
    loss = 0.0;
    num_classes = max(label(:));
    
    for i = 1 : num_train %for each training example
        
        x_i = data(:, i);
        y_i = label(i);
        dW = W;
        dW(:, y_i) = 0; 
        [M1,y_hat]=max(dW' * x_i);
        
        [loss, W] = compute_hingeLoss(W, y_hat, x_i, y_i, C, learningRate, num_train, num_classes, loss, i);
    end
    
    sumW = sum(vecnorm(W).^2);
end


