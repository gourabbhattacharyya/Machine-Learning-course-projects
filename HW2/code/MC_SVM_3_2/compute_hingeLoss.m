function [ loss, W ] = compute_hingeLoss(W, y_hat, x, y, C, learningRate, num_train, num_classes, loss, iter)
%compute hinge loss
lossVal = W(:, y_hat)' * x - W(:, y)' * x + 1;
        
        for j = 1:num_classes
            if lossVal > 0
                if( j == y)  %y_i
                    W(:,j) = W(:,j) - learningRate * ((W(:, y))./num_train  - C .* x);
                elseif(j == y_hat) %y_i_hat
                    W(:,j) = W(:,j) - learningRate * ((W(:, y_hat))./num_train  + C .* x);
                else
                    W(:,j) = W(:,j) - learningRate * (W(:, j))./num_train;
                end    
            else
                W(:,j) = W(:,j) - learningRate * (W(:, j))./num_train;
            end
      
        end
        
        if (mod(iter, 100) == 0)
                fprintf('Iteration %d / %d: loss %f \n', iter, num_train, lossVal);
            
        end
        
        lossVal = max(lossVal, 0); % max loss
        loss = loss + (sum(vecnorm(W).^2))/(2 * num_train) + C * lossVal;
end