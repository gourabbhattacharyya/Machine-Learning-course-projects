function [ out ] = normalize( data )
% Normalize the data
% Rescale the feature values into the range of [-1 1]
% B = +1, M = -1
    tmp = data';

    mean_tmp = mean(tmp);
    tmp = bsxfun(@minus, tmp, mean_tmp);
    tmp = bsxfun(@rdivide, tmp, std(tmp));

    out = tmp';
end
