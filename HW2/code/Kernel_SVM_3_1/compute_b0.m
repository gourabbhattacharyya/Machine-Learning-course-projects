function [ b0 , nSV] = compute_b0( desired, alpha, K, C, threshold , nSV)

    sv_index = find(alpha > threshold * max(alpha));
    
    % number of support vectors
    nSV = [nSV, size(sv_index, 1)];
        b = zeros(length(sv_index),1);
        for i = 1:length(b)
            b(i) = desired(sv_index(i)) - (alpha.*desired)' * K(sv_index(i),:)';
        end
        b0 = mean(b);  
end

