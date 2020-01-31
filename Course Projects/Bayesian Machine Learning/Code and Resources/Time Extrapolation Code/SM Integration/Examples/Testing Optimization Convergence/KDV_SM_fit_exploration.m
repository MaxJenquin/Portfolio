% @author: Maxwell Jenquin, based on code by Maziar Raissi

% utility to explore the fit of the SM model with respect to rng

function error = KDV_SM_fit_exploration(nmix)
%% Setup
addpath ../../
addpath ../../Utilities
addpath ../../Kernels/KDV
addpath ../../Utilities/export_fig

function CleanupFun()
    rmpath ../../
    rmpath ../../Utilities
    rmpath ../../Kernels/KDV
    rmpath ../../Utilities/export_fig
end

finishup = onCleanup(@() CleanupFun());

set(0,'defaulttextinterpreter','latex')

%% Load Data, Initialize
load('../../Data/kdv.mat','usol','t','x')
u_star = real(usol); %512x201
t_star = t; %201x1
x_star = x'; %512x1
N_star = size(x_star,1);
nsteps = size(t_star,1)-1;

% number of sample points for the timesteps sampled
N0 = 80;
N1 = 80;

ncoeffs = 2; % KdV has 2 coefficients to learn

% randomize initialization, since that's what we're testing
rng('shuffle'); 
i=randi(nsteps);
dt = t_star(i+1)-t_star(i);

% sample from simulation data as ground truth
idx0 = randsample(N_star, N0);
x0 = x_star(idx0,:);
u0 = u_star(idx0,i);

idx1 = randsample(N_star, N1);
x1 = x_star(idx1,:);
u1 = u_star(idx1,i+1);

% initialize model
hyp = [0 rand(1,ncoeffs)];
model = SM_HPM_extrap(x1,u1,x0,u0,dt,hyp,nmix,ncoeffs);
model = model.train(500, 20);

% compute relative error in prediction, measuring fit to "ground truth"
[pred_n_star, ~] = model.predict(x_star);
error = norm(pred_n_star - u_star(:,i+1))/norm(u_star(:,i+1));

end