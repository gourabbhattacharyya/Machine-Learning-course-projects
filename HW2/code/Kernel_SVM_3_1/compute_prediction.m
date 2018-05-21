function [ pred ] = compute_prediction(size_data, train_label, alpha, b0, K)

pred = sum(bsxfun(@times, K, (alpha .* train_label)') , 2) + b0 * ones(size_data, 1);

end

