function hyps = KDV_test()

%%
clc; close all;

addpath ..
addpath ../Utilities
addpath ../Kernels/KDV
addpath ../Utilities/export_fig

function CleanupFun()
    rmpath ..
    rmpath ../Utilities
    rmpath ../Kernels/KDV
    rmpath ../Utilities/export_fig
end

finishup = onCleanup(@() CleanupFun());

set(0,'defaulttextinterpreter','latex')

%% Load Data, Setup
load('../Data/kdv.mat','usol','t','x')
u_star = real(usol); %512x201
t_star = t; %201x1
x_star = x'; %512x1
domain = [min(x_star),max(x_star)];
% rescale domain to [0,1] for numerical stability of high order polynomials
% in tau = x-x'
x_star_resc = (x_star - domain(1))./(domain(2)-domain(1));
N_star = size(x_star_resc,1);
nsteps = size(t_star,1)-1;

N0 = 10;
N1 = 10;

nmix = 3;
ncoeffs = 2; %KdV has 2 coefficients

%% Clean Data
rng('shuffle');
i=randi(nsteps);
dt = t_star(i+1)-t_star(i);

idx0 = randsample(N_star, N0);
x0 = x_star_resc(idx0,:);
u0 = u_star(idx0,i);

idx1 = randsample(N_star, N1);
x1 = x_star_resc(idx1,:);
u1 = u_star(idx1,i+1);

% hyperparameters ordered as such: 
        % [ logsigma, weights, means, logvariances, coefficients ] 
        % logsigma of size 1, weights, means and logvariances of size nmix
        % coefficients vary in number by particular PDE - here we have 2.
hyp = [0 rand(1,3*nmix+ncoeffs)];
model = SM_HPM_extrap(x1,u1,x0,u0,dt,hyp,nmix,ncoeffs);
model = model.train(1000);

hyps = model.hyp;
params = model.hyp(end-ncoeffs+1:end);

[pred_n_star, var_n_star] = model.predict(x_star);
var_n_star = abs(diag(var_n_star));

error = norm(pred_n_star - u_star(:,i+1))/norm(u_star(:,i+1));

fprintf(1,'=========================\n');
fprintf(1,'Step: %d, Time = %.2f\n\nNLML = %.2f, Error = %.2e\n\n', i, ...
    t_star(i+1), model.NLML, error);

str = sprintf('%.4f  ', params);
fprintf('Parameters: %s\n\n', str)
fprintf(1,'=========================\n\n');


figure();
plot_prediction_1D(x_star, u_star(:,i+1), pred_n_star, var_n_star, ...
    '$x$', '$u(t,x)$', 'Prediction (clean data)');

drawnow;


end