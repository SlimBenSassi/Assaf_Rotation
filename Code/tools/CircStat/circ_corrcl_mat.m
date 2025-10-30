function [rho, pval] = circ_corrcl_mat(alpha, x)
%
% [rho pval ts] = circ_corrcc(alpha, x)
%   Correlation coefficient between a matrix of circular vairables and one linear random
%   variable.
%
%   Input:
%     alpha   matrix of angles in radians
%     x       sample of linear random variable
%
%   Output:
%     rho     correlation coefficient
%     pval    p-value
%
%   Adapted from circ_corrcl from circ_stat (Berens, 2009)


if size(x,1)==1
    x=x';
end

if size(alpha,1) ~=length(x)
    if size(alpha,2) ==length(x)
        alpha = alpha';
    else
        error('size of angle matrix must match length of linear vector')
    end
end

n = length(x);

% compute correlation coefficent for sin and cos independently
rxs = corr(x,sin(alpha));
rxc = corr(x,cos(alpha));
rcs = diag(corr(sin(alpha),cos(alpha)))';

% compute angular-linear correlation (equ. 27.47)
rho = sqrt((rxc.^2 + rxs.^2 - 2*rxc.*rxs.*rcs)./(1-rcs.^2));

% compute pvalue
pval = 1 - chi2cdf(n*rho.^2,2);

