function K = k00_sm( x, xp, hyp, nmix, ubar, ubarp, dt )
% Computes the covariance between the n-1st timestep and itself, via
% spectral mixture kernel for the KdV equation

% get weights, means, variances, there are nmix of each.
weights = exp(hyp(1:nmix));
means = exp(hyp(nmix+1:2*nmix));
% variances are actually logs of variances to maintain positivity, so:
vars = exp(hyp(2*nmix+1:3*nmix));
% get coefficients
lambda = hyp(3*nmix+1:end);

%tau matrix
n_x = size(x,1);
n_xp = size(xp,1);
tau = repmat(x,1,n_xp)-repmat(xp',n_x,1);

ubar = repmat(ubar,1,n_xp);
ubarp = repmat(ubarp',n_x,1);

K = zeros(n_x,n_xp);
tau_2 = tau.^2;
tau_3 = tau_2.*tau;
tau_4 = tau_3.*tau;
tau_5 = tau_4.*tau;
tau_6 = tau_5.*tau;
for i=1:nmix
    a = -2*vars(i)*(pi^2);
    b = 2*pi*means(i);
    %first derivative coefficients
    d1k_cos = p1c(a,b); % tau
    d1k_sin = p1s(a,b); % tau^0
    %second derivative coefficients
    d2k_cos = p2c(a,b); % tau^2, tau^0
    d2k_sin = p2s(a,b); % tau
    %fourth derivative coefficients
    d4k_cos = p4c(a,b); % tau^4, tau^2, tau^0
    d4k_sin = p4s(a,b); % tau^3, tau
    %sixth derivative coefficients
    d6k_cos = p6c(a,b); % tau^6, tau^4, tau^2, tau^0
    d6k_sin = p6s(a,b); % tau^5, tau^3, tau
    
    %precompute
    expterm = exp(a.*tau_2);
    costerm = cos(b.*tau);
    sinterm = sin(b.*tau);
    %compute contributions from each derivative of k
    d0k_contr = expterm.*costerm;
    d1k_contr = (dt*lambda(1)).*(ubar - ubarp).*expterm.*(d1k_cos.*tau.*costerm...
        +d1k_sin.*sinterm);
    d2k_contr = -(dt*lambda(1))^2.*ubar.*ubarp.*expterm.*((d2k_cos(1).*tau_2...
        +d2k_cos(2)).*costerm + (d2k_sin(1).*tau).*sinterm);
    d4k_contr = -(dt^2)*lambda(1)*lambda(2)*(ubar+ubarp).*expterm.*((d4k_cos(1)...
        .*tau_4+d4k_cos(2).*tau_2+d4k_cos(3)).*costerm + (d4k_sin(1).*tau_3...
        +d4k_sin(2).*tau).*sinterm);
    d6k_contr = -(dt*lambda(2))^2.*expterm.*((d6k_cos(1).*tau_6 + d6k_cos(2).*tau_4...
        +d6k_cos(3).*tau_2+d6k_cos(4)).*costerm + (d6k_sin(1).*tau_5 +...
        d6k_sin(2).*tau_3 + d6k_sin(3).*tau).*sinterm);
    K = K + weights(i).*(d0k_contr+d1k_contr+d2k_contr+d4k_contr+d6k_contr);
end

end
