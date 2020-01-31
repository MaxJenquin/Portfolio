function K = k10_sm( x, xp, hyp, nmix, ubarp, dt )
% Computes the covariance between the nth and n-1st timestep, via spectral
% mixture kernel for the KdV equation

% get weights, means, variances, there are nmix of each.
weights = exp(hyp(1:nmix));
means = exp(hyp(nmix+1:2*nmix));
% variances are stored as logs of variances to maintain positivity, so:
vars = exp(hyp(2*nmix+1:3*nmix));
% get coefficients
lambda = hyp(1,3*nmix+1:end);

% tau matrix
n_x = size(x,1);
n_xp = size(xp,1);
tau = repmat(x,1,n_xp)-repmat(xp',n_x,1);

ubarp = repmat(ubarp',n_x,1);

K = zeros(n_x,n_xp);
tau_2 = tau.^2;
tau_3 = tau.^3;
for i=1:nmix
    a = -2*vars(i)*(pi^2);
    b = 2*pi*means(i);
    %first derivative coefficients
    d1k_cos = p1c(a,b); % tau^1
    d1k_sin = p1s(a,b); % tau^0
    %third derivative coefficients
    d3k_cos = p3c(a,b); % tau^3, tau^1
    d3k_sin = p3s(a,b); % tau^2, tau^0
    
    %precompute
    expterm = exp(a.*tau_2);
    costerm = cos(b.*tau);
    sinterm = sin(b.*tau);
    %compute contributions from each derivative of k
    d0k_contr = expterm.*costerm;
    d1k_contr = (-dt*lambda(1)).*ubarp.*expterm.*(d1k_cos.*tau.*costerm...
        +d1k_sin.*sinterm);
    d3k_contr = (dt*lambda(2)).*expterm.*((d3k_cos(1).*tau_3+d3k_cos(2).*tau)...
        .*costerm + (d3k_sin(1).*tau_2+d3k_sin(2).*tau).*sinterm);
    K = K + weights(i) * (d0k_contr + d1k_contr + d3k_contr);
end

end

