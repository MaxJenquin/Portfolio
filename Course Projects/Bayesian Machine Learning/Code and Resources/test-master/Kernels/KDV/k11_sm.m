function K = k11_sm( x, xp, lambda, SMhyp, nmix, flag)
% Measures covariance of nth timestep with itself, via spectral mixture
% kernel for the KdV equation (flag = 0). For nonzero flag values, returns
% derivative of this portion of the kernel wrt parameters lambda.

switch flag
    case 0
        % no derivative, kernel evaluation
        % get weights, means, variances, there are nmix of each.
        weights = exp(SMhyp(1:nmix));
        means = exp(SMhyp(nmix+1:2*nmix));
        vars = exp(SMhyp(2*nmix+1:3*nmix)).^2;


        % tau matrix
        n_x = size(x,1);
        n_xp = size(xp,1);
        tau = repmat(x,1,n_xp) - repmat(xp',n_x,1);

        K = zeros(n_x,n_xp);
        for i=1:nmix
            a = -2*vars(i)*(pi^2);
            b = 2*pi*means(i);
            K = K + weights(i).*exp(a.*tau.^2).*cos(b.*tau);
            %note no lambda dependence in this portion of the kernel!
        end
    case 1
        % derivative wrt lambda 1 (no lambda dependence in this portion of
        % kernel)
        n_x = size(x,1);
        n_xp = size(xp,1);
        K = zeros(n_x,n_xp);
    case 2
        % derivative wrt lambda 2 (no lambda dependence in this portion of
        % the kernel)
        n_x = size(x,1);
        n_xp = size(xp,1);
        K = zeros(n_x,n_xp);
    otherwise
        disp('Must set flag to recieve output - k11_SM');
end


end

