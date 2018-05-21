function [ prediction ] = compute_prediction(data, W)

y_pred = zeros(size(data, 1), 1);
scores = data' * W;
[M, y_pred] = max(scores, [], 2);
prediction = y_pred;

end

