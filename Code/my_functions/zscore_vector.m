function z_out = zscore_vector(x_in)
% ZSCORE_VECTOR Calculates the Z-score for a vector.
% Inputs: x_in (N x 1 vector)
% Outputs: z_out (N x 1 Z-scored vector)

    m = mean(x_in);
    s = std(x_in);
    
    % If STD is zero (data is constant), set STD to 1 to avoid division by zero.
    if s == 0
        s = 1;
    end
    
    z_out = (x_in - m) / s;
end