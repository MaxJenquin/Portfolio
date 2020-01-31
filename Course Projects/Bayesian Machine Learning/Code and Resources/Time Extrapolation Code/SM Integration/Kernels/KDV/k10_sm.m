function K = k10_sm( x, xp, lambda, SMhyp, nmix, ubarp, dt, flag )
% Computes the covariance between the nth and n-1st timestep, via spectral
% mixture kernel for the KdV equation (flag = 0). For nonzero flag values, 
% returns derivative of this portion of the kernel wrt parameters lambda.

% get weights, means, variances, there are nmix of each.
weights = exp(SMhyp(1:nmix));
means = exp(SMhyp(nmix+1:2*nmix));
vars = exp(SMhyp(2*nmix+1:3*nmix)).^2;

% tau matrix
n_x = size(x,1);
n_xp = size(xp,1);
tau = repmat(x,1,n_xp)-repmat(xp',n_x,1);

ubarp = repmat(ubarp',n_x,1);

K = zeros(n_x,n_xp);
tau_2 = tau.^2;
tau_3 = tau.^3;

switch flag
    case 0
        % no derivative, kernel evaluation
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
    case 1
        % derivative wrt lambda 1
        for i=1:nmix
            a = -2*vars(i)*(pi^2);
            b = 2*pi*means(i);
            %first derivative coefficients
            d1k_cos = p1c(a,b); % tau^1
            d1k_sin = p1s(a,b); % tau^0
            
            %precompute
            expterm = exp(a.*tau_2);
            costerm = cos(b.*tau);
            sinterm = sin(b.*tau);
            
            d1k_contr = -dt.*ubarp.*expterm.*(d1k_cos.*tau.*costerm...
                +d1k_sin.*sinterm);
            K = K + weights(i) * d1k_contr;
        end
    case 2
        % derivative wrt lambda 2
        for i=1:nmix
            a = -2*vars(i)*(pi^2);
            b = 2*pi*means(i);
            %third derivative coefficients
            d3k_cos = p3c(a,b); % tau^3, tau^1
            d3k_sin = p3s(a,b); % tau^2, tau^0
            
            %precompute
            expterm = exp(a.*tau_2);
            costerm = cos(b.*tau);
            sinterm = sin(b.*tau);
            
            d3k_contr = dt.*expterm.*((d3k_cos(1).*tau_3+d3k_cos(2).*tau)...
                .*costerm + (d3k_sin(1).*tau_2+d3k_sin(2).*tau).*sinterm);
            K = K + weights(i) * d3k_contr;
        end
    otherwise
        disp('Must set flag to recieve output - k10_SM');
end




end

