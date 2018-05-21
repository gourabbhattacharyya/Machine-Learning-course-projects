function [ prediction, labelFinal ] = compute_prediction(data, W, label)

y_pred = zeros(size(data, 1), 1);
scores = data' * W;
[M, y_pred] = max(scores, [], 2);
prediction = y_pred;

labelFinal = label;
for i = 1 : size(data, 2)
    if label(i) == -1
        labelFinal(i) = 2;
    end
end

end