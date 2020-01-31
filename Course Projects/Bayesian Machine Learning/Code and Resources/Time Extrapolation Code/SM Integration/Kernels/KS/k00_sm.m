function K = k00_sm( x, xp, lambda, SMhyp, nmix, ubar, ubarp, dt, flag )
% Computes the covariance between the n-1st timestep and itself, via
% spectral mixture kernel for the KS equation (flag = 0). For nonzero 
% flag values, returns derivative of this portion of the kernel wrt 
% parameters lambda.

% get weights, means, variances, there are nmix of each.
weights = exp(SMhyp(1:nmix));
means = exp(SMhyp(nmix+1:2*nmix));
vars = exp(SMhyp(2*nmix+1:3*nmix)).^2;

%tau matrix
n_x = size(x,1);
n_xp = size(xp,1);
tau = repmat(x,1,n_xp)-repmat(xp',n_x,1);

%precompute
ubar = repmat(ubar,1,n_xp);
ubarp = repmat(ubarp',n_x,1);
uprod = ubar.*ubarp;
udiff = ubar-ubarp;

K = zeros(n_x,n_xp);
tau_2 = tau.^2;
tau_3 = tau_2.*tau;
tau_4 = tau_3.*tau;
tau_5 = tau_4.*tau;
tau_6 = tau_5.*tau;
tau_7 = tau_6.*tau;
tau_8 = tau_7.*tau;

switch flag
    case 0
        % no derivative, kernel evaluation
        for i=1:nmix
            a = -2*vars(i)*(pi^2);
            b = 2*pi*means(i);
            %first derivative coefficients
            d1k_cos = p1c(a,b); % tau
            d1k_sin = p1s(a,b); % tau^0
            %second derivative coefficients
            d2k_cos = p2c(a,b); % tau^2, tau^0
            d2k_sin = p2s(a,b); % tau
            %third derivative coefficients
            d3k_cos = p3c(a,b); % tau^3, tau
            d3k_sin = p3s(a,b); % tau^2, tau^0
            %fourth derivative coefficients
            d4k_cos = p4c(a,b); % tau^4, tau^2, tau^0
            d4k_sin = p4s(a,b); % tau^3, tau
            %fifth derivative coefficients
            d5k_cos = p5c(a,b); % tau^5, tau^3, tau
            d5k_sin = p5s(a,b); % tau^4, tau^2, tau^0
            %sixth derivative coefficients
            d6k_cos = p6c(a,b); % tau^6, tau^4, tau^2, tau^0
            d6k_sin = p6s(a,b); % tau^5, tau^3, tau
            %eigth derivative coefficients
            d8k_cos = p8c(a,b); % tau^8, tau^6, tau^4, tau^2, tau^0
            d8k_sin = p8s(a,b); % tau^7, tau^5, tau^3, tau
            %precompute
            expterm = exp(a.*tau_2);
            costerm = cos(b.*tau);
            sinterm = sin(b.*tau);
            %compute contributions from each derivative of k
            d0k_contr = expterm.*costerm;
            d1k_contr = (dt*lambda(1)).*udiff.*expterm.*(d1k_cos.*tau.*costerm...
                +d1k_sin.*sinterm);
            d2k_contr = (2*dt*lambda(2)-(dt*lambda(1))^2.*uprod).*expterm.*((d2k_cos(1).*tau_2...
                +d2k_cos(2)).*costerm + (d2k_sin(1).*tau).*sinterm);
            d3k_contr = (dt^2*lambda(1)*lambda(2)*udiff).*expterm.*((d3k_cos(1).*tau_3...
                +d3k_cos(2).*tau).*costerm + d3k_sin(1).*tau_2 + d3k_sin(2).*sinterm);
            d4k_contr = (2*dt*lambda(3)+(dt*lambda(2))^2).*expterm.*((d4k_cos(1)...
                .*tau_4+d4k_cos(2).*tau_2+d4k_cos(3)).*costerm + (d4k_sin(1).*tau_3...
                +d4k_sin(2).*tau).*sinterm);
            d5k_contr = (dt^2)*lambda(1)*lambda(2)*udiff.*expterm.*((d5k_cos(1).*tau_5...
                +d5k_cos(2).*tau_3+d5k_cos(3).*tau).*costerm + (d5k_sin(1).*tau_4...
                +d5k_sin(2).*tau_2+d5k_sin(3)).*sinterm);
            d6k_contr = 2*(dt^2)*lambda(2)*lambda(3).*expterm.*((d6k_cos(1).*tau_6...
                + d6k_cos(2).*tau_4+d6k_cos(3).*tau_2+d6k_cos(4)).*costerm...
                + (d6k_sin(1).*tau_5 +d6k_sin(2).*tau_3 + d6k_sin(3).*tau).*sinterm);
            d8k_contr = (dt*lambda(3))^2.*expterm.*((d8k_cos(1).*tau_8+d8k_cos(2).*tau_6...
                +d8k_cos(3).*tau_4+d8k_cos(4).*tau_2+d8k_cos(5)).*costerm...
                +(d8k_sin(1).*tau_7+d8k_sin(2).*tau_5+d8k_sin(3).*tau_3...
                +d8k_sin(4).*tau).*sinterm);
            K = K + weights(i).*(d0k_contr+d1k_contr+d2k_contr+d3k_contr...
                +d4k_contr+d5k_contr+d6k_contr+d8k_contr);
        end
    case 1
        % derivative wrt lambda 1
        for i=1:nmix
            a = -2*vars(i)*(pi^2);
            b = 2*pi*means(i);
            %first derivative coefficients
            d1k_cos = p1c(a,b); % tau
            d1k_sin = p1s(a,b); % tau^0
            %second derivative coefficients
            d2k_cos = p2c(a,b); % tau^2, tau^0
            d2k_sin = p2s(a,b); % tau
            %third derivative coefficients
            d3k_cos = p3c(a,b); % tau^3, tau
            d3k_sin = p3s(a,b); % tau^2, tau^0
            %fifth derivative coefficients
            d5k_cos = p5c(a,b); % tau^5, tau^3, tau
            d5k_sin = p5s(a,b); % tau^4, tau^2, tau^0
            %precompute
            expterm = exp(a.*tau_2);
            costerm = cos(b.*tau);
            sinterm = sin(b.*tau);
            %compute contributions from each derivative of k
            d1k_contr = dt.*udiff.*expterm.*(d1k_cos.*tau.*costerm...
                +d1k_sin.*sinterm);
            d2k_contr = -2*(dt^2)*lambda(1).*uprod.*expterm.*((d2k_cos(1).*tau_2...
                +d2k_cos(2)).*costerm + (d2k_sin(1).*tau).*sinterm);
            d3k_contr = (dt^2)*lambda(2)*udiff.*expterm.*((d3k_cos(1)...
                .*tau_3+d3k_cos(2).*tau).*costerm + (d3k_sin(1).*tau_2...
                +d3k_sin(2)).*sinterm);
            d5k_contr = (dt^2)*lambda(3)*udiff.*expterm.*((d5k_cos(1).*tau_5...
                +d5k_cos(2).*tau_3+d5k_cos(3).*tau).*costerm + (d5k_sin(1).*tau_4...
                +d5k_sin(2).*tau_2+d5k_sin(3)).*sinterm);
            K = K + weights(i).*(d1k_contr+d2k_contr+d3k_contr+d5k_contr);
        end
    case 2
        % derivative wrt lambda 2
        for i=1:nmix
            a = -2*vars(i)*(pi^2);
            b = 2*pi*means(i);
            %second derivative coefficients
            d2k_cos = p2c(a,b); % tau^2, tau^0
            d2k_sin = p2s(a,b); % tau
            %third derivative coefficients
            d3k_cos = p3c(a,b); % tau^3, tau
            d3k_sin = p3s(a,b); % tau^2, tau^0
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
            d2k_contr = 2*dt*expterm.*((d2k_cos(1).*tau_2+d2k_cos(2)).*costerm...
                + (d2k_sin(1).*tau).*sinterm);
            d3k_contr = (dt^2)*lambda(1)*udiff.*expterm.*((d3k_cos(1)...
                .*tau_3+d3k_cos(2).*tau).*costerm + (d3k_sin(1).*tau_2...
                +d3k_sin(2)).*sinterm);
            d4k_contr = 2*(dt^2)*lambda(2).*expterm.*((d4k_cos(1).*tau_4...
                +d4k_cos(2).*tau_2+d4k_cos(3)).*costerm + (d4k_sin(1).*tau_3...
                +d4k_sin(2).*tau).*sinterm);
            d6k_contr = 2*(dt^2)*lambda(3).*expterm.*((d6k_cos(1).*tau_6...
                + d6k_cos(2).*tau_4+d6k_cos(3).*tau_2+d6k_cos(4)).*costerm...
                + (d6k_sin(1).*tau_5 +d6k_sin(2).*tau_3 + d6k_sin(3).*tau).*sinterm);
            K = K + weights(i).*(d2k_contr+d3k_contr+d4k_contr+d6k_contr);
        end
    case 3
        % derivative wrt lambda 3
        for i=1:nmix
            a = -2*vars(i)*(pi^2);
            b = 2*pi*means(i);
            %fourth derivative coefficients
            d4k_cos = p4c(a,b); % tau^4, tau^2, tau^0
            d4k_sin = p4s(a,b); % tau^3, tau
            %fifth derivative coefficients
            d5k_cos = p5c(a,b); % tau^5, tau^3, tau
            d5k_sin = p5s(a,b); % tau^4, tau^2, tau^0
            %sixth derivative coefficients
            d6k_cos = p6c(a,b); % tau^6, tau^4, tau^2, tau^0
            d6k_sin = p6s(a,b); % tau^5, tau^3, tau
            %eigth derivative coefficients
            d8k_cos = p8c(a,b); % tau^8, tau^6, tau^4, tau^2, tau^0
            d8k_sin = p8s(a,b); % tau^7, tau^5, tau^3, tau
            %precompute
            expterm = exp(a.*tau_2);
            costerm = cos(b.*tau);
            sinterm = sin(b.*tau);
            %compute contributions from each derivative of k
            d4k_contr = 2*dt.*expterm.*((d4k_cos(1).*tau_4+d4k_cos(2).*tau_2...
                +d4k_cos(3)).*costerm + (d4k_sin(1).*tau_3+d4k_sin(2).*tau).*sinterm);
            d5k_contr = (dt^2)*lambda(1).*udiff.*expterm.*((d5k_cos(1).*tau_5...
                +d5k_cos(2).*tau_3+d5k_cos(3).*tau).*costerm + (d5k_sin(1).*tau_4...
                +d5k_sin(2).*tau_2+d5k_sin(3)).*sinterm);
            d6k_contr = 2*(dt^2)*lambda(2).*expterm.*((d6k_cos(1).*tau_6...
                + d6k_cos(2).*tau_4+d6k_cos(3).*tau_2+d6k_cos(4)).*costerm...
                + (d6k_sin(1).*tau_5 +d6k_sin(2).*tau_3 + d6k_sin(3).*tau).*sinterm);
            d8k_contr = 2*(dt^2)*lambda(3).*expterm.*((d8k_cos(1).*tau_8+...
                d8k_cos(2).*tau_6+d8k_cos(3).*tau_4+d8k_cos(4).*tau_2...
                +d8k_cos(5)).*costerm +(d8k_sin(1).*tau_7+d8k_sin(2).*tau_5...
                +d8k_sin(3).*tau_3+d8k_sin(4).*tau).*sinterm);
            K = K + weights(i).*(d4k_contr+d5k_contr+d6k_contr+d8k_contr);
        end
    otherwise
        disp('Must set flag to recieve output - k00_SM');
end

end

