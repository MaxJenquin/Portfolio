% @author: Maxwell Jenquin

classdef SM_HPM_extrap
    properties
        dt % timestep size
        X1, U1 % data at time n
        X0, U0 % data at time n-1
        nmix % number of Gaussians composing spectral density of kernel
        ncoeffs % number of coefficients in equation (for bookkeeping)
        hyp % hyperparameters
        % hyperparameters ordered as such: 
        % [ logsigma, logweights, logmeans, logvariances, coefficients ] 
        % logsigma of size 1, logweights, logmeans and logvariances of size
        % nmix, coefficients vary in number by particular PDE
        NLML % negative log marginal likelihood
    end
    
    methods
        function obj = SM_HPM_extrap(X1, U1, X0, U0, dt, hyp, nmix, ncoeffs)
            obj.X1 = X1;
            obj.U1 = U1;
            obj.X0 = X0;
            obj.U0 = U0;
            obj.dt = dt;
            obj.hyp = hyp;
            obj.nmix = nmix;
            obj.ncoeffs = ncoeffs;
            fprintf('Total number of parameters: %d\n', length(obj.hyp));
        end
        
        function NLML = likelihood(obj, hyp)
            y = [obj.U1(:); obj.U0(:)];
            N = size(y,1);
            
            X1_ = obj.X1;
            X0_ = obj.X0;
            U0_ = obj.U0;
            dt_ = obj.dt;
            
            sigma = exp(hyp(1));
            hyp_ = hyp(2:end);
            
            K11 = k11_sm(X1_, X1_, hyp_, obj.nmix);
            K10 = k10_sm(X1_, X0_, hyp_, obj.nmix, U0_, dt_);
            K00 = k00_sm(X0_, X0_, hyp_, obj.nmix, U0_, U0_, dt_);
            
            K = [K11 K10;
                K10' K00];
            
            L = jit_chol(K+sigma*eye(N));
            alpha = L'\(L\y);
            NLML = 0.5*y'*alpha + sum(log(diag(L))) + log(2*pi)*N/2.0;
        end
        
        function obj = train(obj, n_evals)
            nmix_ = obj.nmix;
            ncoeffs_ = obj.ncoeffs;
            
            % lognoise bounds: [-5, 5]
            
            % logweight bounds: [-5, 5] loosen if necessary
            
            % logmean bounds: [-5, 2.5] avoids high frequencies
            
            % logvariance bounds: [-5, 5] keeps variance of each mixture 
            % component between machine precision and about 150
            
            % coefficient bounds: [-20, 20] arbitrary, tighten if necessary
            
            lower = [-5, -5*ones(1,nmix_), -5*ones(1,nmix_),...
                -5*ones(1,nmix_), -20*ones(1,ncoeffs_)];
            upper = [5, 5*ones(1,nmix_), 2.5*ones(1,nmix_),...
                5*ones(1,nmix_), 20*ones(1,ncoeffs_)];
            
            % surrogateopt requires global optimization package for MatLab
            options = optimoptions('surrogateopt','MaxFunctionEvaluations'...
                ,n_evals,'Display','off');
            
            [obj.hyp, obj.NLML] = surrogateopt(@obj.likelihood,lower,upper,options);
            fprintf('Model NLML:\n');
            disp(obj.NLML);
            
        end
        
        function obj = train_mincon(obj,n_evals)
            nmix_ = obj.nmix;
            ncoeffs_ = obj.ncoeffs;
            
            lower = [-5, -5*ones(1,nmix_), -5*ones(1,nmix_),...
                -5*ones(1,nmix_), -20*ones(1,ncoeffs_)];
            upper = [5, 5*ones(1,nmix_), 2.5*ones(1,nmix_),...
                5*ones(1,nmix_), 20*ones(1,ncoeffs_)];
            
            options = optimoptions('fmincon','MaxFunctionEvaluations'...
                ,n_evals);%,'Display','off');
            [x, fval] = fmincon(@obj.likelihood,obj.hyp,[],[],[],[],lower,upper,[],options);
            obj.hyp = x;
            obj.NLML = fval;
            fprintf('Model NLML:\n');
            disp(obj.NLML);
        end
        
        function [u1_star_mean, u1_star_var] = predict(obj, X1_star)
                        
            y = [obj.U1(:); obj.U0(:)];
            
            X1_ = obj.X1;
            X0_ = obj.X0; U0_ = obj.U0;
            dt_ = obj.dt;
            
            sigma = exp(obj.hyp(1));
            hyp_ = obj.hyp(2:end);
            
            N = size(y,1);
            
            K11 = k11_sm(X1_, X1_, hyp_, obj.nmix);
            K10 = k10_sm(X1_, X0_, hyp_, obj.nmix, U0_, dt_);
            K00 = k00_sm(X0_, X0_, hyp_, obj.nmix, U0_, U0_, dt_);
            
            K = [K11  K10;
                 K10' K00];
            
            % Cholesky factorisation
            L = jit_chol(K + sigma*eye(N));
            
            K11 = k11_sm(X1_star, X1_, hyp_, obj.nmix);
            K10 = k10_sm(X1_star, X0_, hyp_, obj.nmix, U0_, dt_);
            
            psi = [K11 K10];
            
            u1_star_mean = psi*(L'\(L\y));
            
            alpha = (L'\(L\psi'));
            
            u1_star_var = k11_sm(X1_star, X1_star, hyp_, obj.nmix) - psi*alpha;
        end
        
        function [next_mean, next_var] = extrap_predict(obj, oldx, oldu, newx)
            % get necessary variables
            dt_= obj.dt;
            sigma = exp(obj.hyp(1));
            hyp_ = obj.hyp(2:end);
            oldu_ = oldu(:);
            N = size(oldu_,1);
            
            % evaluate covariances
            K11 = k11_sm(newx, newx, hyp_, obj.nmix);
            K10 = k10_sm(newx, oldx, hyp_, obj.nmix, oldu, dt_);
            K00 = k00_sm(oldx, oldx, hyp_, obj.nmix, oldu, oldu, dt_);
            
            % add pointwise noise estimate - note we use the global model
            % noise estimate to maintain tight predictions here. Future
            % work towards accurate variance estimates coming.
            L = jit_chol(K00+sigma*eye(N));

            % evaluate predictive distribution
            next_mean = K10*(L'\(L\oldu_));
            next_var = diag(K11 + K10*(L'\(L\K10')));
            
        end
    end
end