addpath ..
addpath ../Utilities
addpath ../Kernels/KDV
addpath ../Utilities/export_fig

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

hypsimple = initSMhypers(nmix,x1,u1);
hypadv = initSMhypersadvanced(nmix,x1,u1,1);