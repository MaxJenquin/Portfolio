function K = k11_sm( x, xp, hyp, nmix )
% Measures covariance of nth timestep with itself, via spectral mixture
% kernel for the KdV equation.

% get weights, means, variances, there are nmix of each.
weights = exp(hyp(1:nmix));
means = exp(hyp(nmix+1:2*nmix));
% variances are actually logs of variances to maintain positivity, so:
vars = exp(hyp(2*nmix+1:3*nmix));
% get coefficients
lambda = hyp(3*nmix+1:end);

% tau matrix
n_x = size(x,1);
n_xp = size(xp,1);
tau = repmat(x,1,n_xp) - repmat(xp',n_x,1);

K = zeros(n_x,n_xp);
for i=1:nmix
    a = -2*vars(i)*(pi^2);
    b = 2*pi*means(i);
    K = K + weights(i).*exp(a.*tau.^2).*cos(b.*tau);
    %note no lambda dependence in this portion of the kernel
end
end

