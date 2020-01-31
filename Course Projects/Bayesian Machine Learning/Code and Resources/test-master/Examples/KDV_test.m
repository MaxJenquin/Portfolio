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
%x_star_resc = (x_star - domain(1))./(domain(2)-domain(1));
x_star_resc = x_star;
N_star = size(x_star_resc,1);
nsteps = size(t_star,1)-1;

N0 = 80;
N1 = 80;

nmix = 4;
ncoeffs = 2; %KdV has 2 coefficients

%% Clean Data
rng('default');
i=randi(nsteps);
dt = t_star(i+1)-t_star(i);

idx0 = randsample(N_star, N0);
x0 = x_star_resc(idx0,:);
u0 = u_star(idx0,i);

idx1 = randsample(N_star, N1);
x1 = x_star_resc(idx1,:);
u1 = u_star(idx1,i+1);

% hyperparameters are log(noise), and coefficients. SM kernel
% hyperparameters are set based on U1's spectrum (see
% lowparam_SM_HPM_extrap.m).
hyp = [0 rand(1,ncoeffs)];
model = lowparam_SM_HPM_extrap(x1,u1,x0,u0,dt,hyp,nmix,ncoeffs);
model = model.train(500, 10);

simdat = u_star(:,(i+2):end);
tdat = t_star((i+2):end);
ext_length = size(t_star, 1) - (i+2);

[unext, varnext] = model.extrap_predict(x1, u1, x_star);
preds = zeros(size(x_star, 1),ext_length); % predictive means
vars = zeros(size(x_star, 1),ext_length); % predictive variances
preds(:, 1) = unext;
vars(:, 1) = varnext;

for k=2:(nsteps-i)
    [temp1, temp2] = model.extrap_predict(x_star, preds(:,k-1), x_star);
    preds(:, k) = temp1;
    vars(:, k) = temp2;
end

fig = figure('Name','Clean Extrapolation Results','NumberTitle','off');
hold on
subplot(3,2,3:4)
plot_surface(tdat, x_star, simdat, '$t$', '$x$', 'Simulation Results');
subplot(3,2,1:2)
plot_surface(tdat, x_star, preds, '$t$', '$x$', 'GP Extrapolation Results');
subplot(3,2,5)
plot_surface(tdat, x_star, abs(preds-simdat), '$t$', '$x$', 'Error Magnitude');
subplot(3,2,6)
plot_surface(tdat, x_star, abs(vars), '$t$', '$x$', 'Variance Magnitude');
%title('KDV Clean Extrapolation vs Simulation','FontWeight','bold');
set(fig, 'units', 'centimeters', 'position', [1 1 21 11]);

hyps = model.hyp;
params = model.hyp(2:end);

[pred_n_star, var_n_star] = model.predict(x_star_resc);
var_n_star = abs(diag(var_n_star));

error = norm(pred_n_star - u_star(:,i+1))/norm(u_star(:,i+1));

fprintf(1,'=========================\n');
fprintf(1,'Step: %d, Time = %.2f\n\nNLML = %.2f, Error = %.2e\n\n', i, ...
    t_star(i+1), model.NLML, error);

str = sprintf('%.4f  ', params);
fprintf('Parameters: %s\n\n', str)
fprintf(1,'=========================\n\n');


figure();
plot_prediction_1D(x_star_resc, u_star(:,i+1), pred_n_star, var_n_star, ...
    '$x$', '$u(t,x)$', 'Prediction (clean data)');
figure();
scatter(x_star_resc(idx1,:), u_star(idx1,i+1));

drawnow;

%% Noisy Data (1%)
noise = 0.01;
rng('default');
u0 = u0 + noise*std(u0)*randn(size(u0));
u1 = u1 + noise*std(u1)*randn(size(u1));

% hyperparameters are log(noise), and coefficients. SM kernel
% hyperparameters are set based on U1's spectrum (see
% lowparam_SM_HPM_extrap.m).
hyp = [0 rand(1,ncoeffs)];
model = lowparam_SM_HPM_extrap(x1,u1,x0,u0,dt,hyp,nmix,ncoeffs);
model = model.train(500, 10);

[unext, varnext] = model.extrap_predict(x1, u1, x_star);
preds = zeros(size(x_star, 1),ext_length); % predictive means
vars = zeros(size(x_star, 1),ext_length); % predictive variances
preds(:, 1) = unext;
vars(:, 1) = varnext;

for k=2:(nsteps-i)
    [temp1, temp2] = model.extrap_predict(x_star, preds(:,k-1), x_star);
    preds(:, k) = temp1;
    vars(:, k) = temp2;
end

fig = figure('Name','Noisy Extrapolation Results','NumberTitle','off');
hold on
subplot(3,2,3:4)
plot_surface(tdat, x_star, simdat, '$t$', '$x$', 'Simulation Results');
subplot(3,2,1:2)
plot_surface(tdat, x_star, preds, '$t$', '$x$', 'GP Extrapolation Results');
subplot(3,2,5)
plot_surface(tdat, x_star, abs(preds-simdat), '$t$', '$x$', 'Error Magnitude');
subplot(3,2,6)
plot_surface(tdat, x_star, abs(vars), '$t$', '$x$', 'Variance Magnitude');
%title('KDV Clean Extrapolation vs Simulation','FontWeight','bold');
set(fig, 'units', 'centimeters', 'position', [1 1 21 11]);

hyp = model.hyp;
params_noise = hyp(2:end);

[pred_n_star, var_n_star] = model.predict(x_star);
var_n_star = abs(diag(var_n_star));

error = norm(pred_n_star - u_star(:,i+1))/norm(u_star(:,i+1));

fprintf(1,'=========================\n');
fprintf(1,'Step: %d, Time = %.2f\n\nNLML = %.2f, Error = %.2e\n\n', i, ...
    t_star(i+1), model.NLML, error);

str = sprintf('%.4f  ', params_noise);
fprintf('Parameters: %s\n\n', str)
fprintf(1,'=========================\n\n');

figure();
plot_prediction_1D(x_star, u_star(:,i+1), pred_n_star, var_n_star, ...
    '$x$', '$u(t,x)$', 'Prediction (noisy data)');

drawnow;


end