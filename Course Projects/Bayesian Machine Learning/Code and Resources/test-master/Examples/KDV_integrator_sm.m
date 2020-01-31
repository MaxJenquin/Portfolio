% @author: Maxwell Jenquin, based on code by Maziar Raissi

function KDV_integrator_sm()

%% Setup
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

%% Load Data, Initialize
load('../Data/kdv.mat','usol','t','x')
u_star = real(usol); %512x201
t_star = t; %201x1
x_star = x'; %512x1
N_star = size(x_star,1);
nsteps = size(t_star,1)-1;

% number of sample points for the timesteps sampled
N0 = 80;
N1 = 80;

nmix = 4; % number of mixture components in base SM kernel
ncoeffs = 2; % KdV has 2 coefficients to learn

% run both clean and noisy simulation?
clean = true;
noisy = true;

%% Clean Data Time Extrapolation
if clean
    rng('default'); % for consistency: other rng settings will give different results
    i=randi(nsteps);
    dt = t_star(i+1)-t_star(i);

    idx0 = randsample(N_star, N0);
    x0 = x_star(idx0,:);
    u0 = u_star(idx0,i);

    idx1 = randsample(N_star, N1);
    x1 = x_star(idx1,:);
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

    % Save results in .mat file to be plotted as vector graphics later
    % Will be saved after running the noisy simulation if noisy is set to true.
    results.x = x_star;
    results.t = tdat;
    results.cleanpreds = preds;
    results.cleanvars = vars;
    results.simdat = simdat;
    if ~noisy
        save('../Results/KDVcleanresults.mat','-struct','results');
    end

    % visualize results
    fig = figure('Name','Clean Extrapolation Results','NumberTitle','off');
    hold on
    subplot(3,2,1:2)
    plot_surface(tdat, x_star, preds, '$t$', '$x$', 'GP Extrapolation Results');
    subplot(3,2,3:4)
    plot_surface(tdat, x_star, simdat, '$t$', '$x$', 'Simulation Results');
    subplot(3,2,5)
    plot_surface(tdat, x_star, abs(preds-simdat), '$t$', '$x$', 'Total Error Magnitude');
    subplot(3,2,6)
    plot_surface(tdat, x_star, abs(vars), '$t$', '$x$', 'Variance Magnitude');
    set(fig, 'units', 'centimeters', 'position', [1 1 41 21]);
    hold off
    
    params = model.hyp(2:end);

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
        '$x$', '$u(t,x)$', 'Full U1 Prediction (clean data)');
    figure();
    scatter(x_star(idx1,:), u_star(idx1,i+1));

    drawnow;
    if ~noisy
        pause('Press Any Key to Continue')
    end
end

%% Time Extrapolation With 1% Noise

noise = 0.01;
rng('default');
% same sample points, now with 1% noise in measurement
u0 = u0 + noise*std(u0)*randn(size(u0));
u1 = u1 + noise*std(u1)*randn(size(u1));

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

% save results
if ~clean
    results.x = x_star;
    results.t = tdat;
    results.noisypreds = preds;
    results.noisyvars = vars;
    results.simdat = simdat;
    save('../Results/KDVnoisyresults.mat','-struct','results');
else
    results.noisypreds = preds;
    results.noisyvars = vars;
    save('../Results/KDVresults.mat','-struct','results');
end

% plot results
fig = figure('Name','Noisy Extrapolation Results','NumberTitle','off');
hold on
subplot(3,2,1:2)
plot_surface(tdat, x_star, preds, '$t$', '$x$', 'GP Extrapolation Results');
subplot(3,2,3:4)
plot_surface(tdat, x_star, simdat, '$t$', '$x$', 'Simulation Results');
subplot(3,2,5)
plot_surface(tdat, x_star, abs(preds-simdat), '$t$', '$x$', 'Error Magnitude');
subplot(3,2,6)
plot_surface(tdat, x_star, abs(vars), '$t$', '$x$', 'Variance Magnitude');
set(fig, 'units', 'centimeters', 'position', [1 1 41 21]);
hold off

params_noise = model.hyp(2:end);

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
    '$x$', '$u(t,x)$', 'Full U1 Prediction (noisy data)');

drawnow;

end