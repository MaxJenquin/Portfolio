% @author: Maxwell Jenquin, modified from HPM class by Maziar Raissi

classdef lowparam_SM_HPM_extrap
    properties
        dt % timestep size
        X1, U1 % data at time n
        X0, U0 % data at time n-1
        nmix % number of Gaussians composing spectral density of kernel
        ncoeffs % number of coefficients in equation (for bookkeeping)
        SMhyp % hyperparameters for the spectral mixture kernel
        % hyperparameters ordered as such:
        % [log(weights), log(means), log(sqrt(variances))]
        % one weight, mean, variance per mixture component, hence nmix of
        % each
        hyp % other model hyperparameters:
        % [ log(noise), coefficients ]
        % one noise level, ncoeffs coefficients (ncoeffs vary by PDE)
        NLML % negative log marginal likelihood
    end
    
    methods
        function obj = lowparam_SM_HPM_extrap(X1, U1, X0, U0, dt, hyp, nmix, ncoeffs)
            obj.X1 = X1;
            obj.U1 = U1;
            obj.X0 = X0;
            obj.U0 = U0;
            obj.dt = dt;
            obj.hyp = hyp;
            obj.nmix = nmix;
            obj.ncoeffs = ncoeffs;
            fprintf('Training SM Kernel Hyperparameters...');
            obj.SMhyp = initSMhypersadvanced(nmix, X1, U1, 1);
            fprintf('Total number of free parameters: %d\n', length(obj.hyp));
        end
        
        function [NLML, D_NLML] = likelihood(obj, hyp)
            y = [obj.U1(:); obj.U0(:)];
            N = size(y,1);
            
            X1_ = obj.X1;
            X0_ = obj.X0;
            U0_ = obj.U0;
            dt_ = obj.dt;
            nmix_ = obj.nmix;
            SMhyp_ = obj.SMhyp;
            
            sigma = exp(hyp(1));
            hyp_ = hyp(2:end);
            
            K11 = k11_sm(X1_, X1_, hyp_, SMhyp_, nmix_, 0);
            K10 = k10_sm(X1_, X0_, hyp_, SMhyp_, nmix_, U0_, dt_, 0);
            K00 = k00_sm(X0_, X0_, hyp_, SMhyp_, nmix_, U0_, U0_, dt_, 0);
            
            K = [K11 K10;
                K10' K00];
            
            L = jit_chol(K+sigma*eye(N));
            alpha = L'\(L\y);
            NLML = 0.5*y'*alpha + sum(log(diag(L))) + log(2*pi)*N/2.0;
            
            D_NLML = 0*hyp;
            Q = L'\(L\eye(N))-alpha*alpha';
            for i=1:obj.ncoeffs
                DK11 = k11_sm(X1_, X1_, hyp_, SMhyp_, nmix_, i);
                DK10 = k10_sm(X1_, X0_, hyp_, SMhyp_, nmix_, U0_, dt_, i);
                DK00 = k00_sm(X0_, X0_, hyp_, SMhyp_, nmix_, U0_, U0_, dt_, i);
                
                DK = [DK11 DK10;
                    DK10' DK00];
                D_NLML(i+1) = sum(sum(Q.*DK))/2;
            end
            D_NLML(1) = sigma*trace(Q)/2;
            
        end
        
        function obj = train(obj, n_iter, n_restarts)
            
            % Train the free parameters repeatedly to obtain a good minimum
            % Note that because some settings result in singular
            % covariance, we restart the trial if that occurs.
            k=1;
            while k<n_restarts
                starthyp = rand(1,1+obj.ncoeffs);
                try
                    [hyp_opt,~,~] = minimize(starthyp, @obj.likelihood, -n_iter);
                    NLML_opt = obj.likelihood(hyp_opt);
                    if k==1
                        best_hyp = hyp_opt;
                        best_NLML = NLML_opt;
                    elseif NLML_opt < best_NLML
                        best_hyp = hyp_opt;
                        best_NLML = NLML_opt;
                    end
                    k = k+1;
                catch
                    disp('Covariance became singular, trying again.')
                end
            end
            % train best result once more for fine tuning
            [obj.hyp,~,~] = minimize(best_hyp, @obj.likelihood, -2*n_iter);
            obj.NLML = obj.likelihood(obj.hyp);
            
        end
        
        function [u1_star_mean, u1_star_var] = predict(obj, X1_star)
                        
            y = [obj.U1(:); obj.U0(:)];
            
            X1_ = obj.X1;
            X0_ = obj.X0; U0_ = obj.U0;
            dt_ = obj.dt;
            SMhyp_= obj.SMhyp;
            
            sigma = exp(obj.hyp(1));
            hyp_ = obj.hyp(2:end);
            
            N = size(y,1);
            
            K11 = k11_sm(X1_, X1_, hyp_, SMhyp_, obj.nmix, 0);
            K10 = k10_sm(X1_, X0_, hyp_, SMhyp_, obj.nmix, U0_, dt_, 0);
            K00 = k00_sm(X0_, X0_, hyp_, SMhyp_, obj.nmix, U0_, U0_, dt_, 0);
            
            K = [K11  K10;
                 K10' K00];
            
            % Cholesky factorisation
            L = jit_chol(K + sigma*eye(N));
            
            K11 = k11_sm(X1_star, X1_, hyp_, SMhyp_, obj.nmix, 0);
            K10 = k10_sm(X1_star, X0_, hyp_, SMhyp_, obj.nmix, U0_, dt_, 0);
            
            psi = [K11 K10];
            
            u1_star_mean = psi*(L'\(L\y));
            
            alpha = (L'\(L\psi'));
            
            u1_star_var = k11_sm(X1_star, X1_star, hyp_, SMhyp_, obj.nmix, 0) - psi*alpha;
        end
        
        function [next_mean, next_var] = extrap_predict(obj, oldx, oldu, newx)
            % get necessary variables
            dt_= obj.dt;
            sigma = exp(obj.hyp(1));
            hyp_ = obj.hyp(2:end);
            SMhyp_ = obj.SMhyp;
            oldu_ = oldu(:);
            N = size(oldu_,1);
            
            % evaluate covariances
            K11 = k11_sm(newx, newx, hyp_, SMhyp_, obj.nmix, 0);
            K10 = k10_sm(newx, oldx, hyp_, SMhyp_, obj.nmix, oldu, dt_, 0);
            K00 = k00_sm(oldx, oldx, hyp_, SMhyp_, obj.nmix, oldu, oldu, dt_, 0);
            
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