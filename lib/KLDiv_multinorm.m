function dist = KLDiv_multinorm(mu1,mu2, sigma1, sigma2)
%  dist = KLDiv_multinorm(mu1,mu2, sigma1, sigma2) Kullback-Leibler
%  divergence of two multinomial Gaussian Distribtions
%  mu1 , sigma1 = mean vector (1 by d) of distributions
%  mu2 , sigma2 = covariance matrix (d by d) of distributions

d = length(mu1);

if size(mu1) ~= size(mu2)
    error('the mean vectors should be the same size');
end

if size(sigma1) ~= size(sigma2)
    error('the covariance matrices should be the same size');
end

dist = 1/2*(log(det(sigma2)/det(sigma1)) ...
            - d ...
            + trace(inv(sigma2) * sigma1) ...
            + (mu2 - mu1) * inv(sigma2) * transpose(mu2 - mu1)...
            );
