function [ alpha ] = compute_alpha( size_data, desired, K, C )
    %% Initialize inputs for quadprog

    H = zeros(size_data, size_data);
    for i = 1:size_data
        for j = 1:size_data
            H(i,j) = desired(i) * desired(j) * K(i,j);
        end
    end
    f = -ones(size_data, 1);
    Aeq = desired';
    Beq = 0;
    lb = zeros(size_data, 1);
    ub = ones(size_data, 1) * C;
    x0 = [];
    options = optimset('Algorithm','interior-point-convex','Display','iter','MaxIter', 10000);
    %optimset('LargeScale', 'off', 'MaxIter', 10000);
    A = [];
    b = [];
    
    %% Call quadprog
    [ alpha, R1, R2 ] = quadprog(H, f, A, b, Aeq, Beq, lb, ub, x0, options);

end


