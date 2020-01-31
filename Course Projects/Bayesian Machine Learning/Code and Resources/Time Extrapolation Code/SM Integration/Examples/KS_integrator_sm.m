% @author: Maxwell Jenquin, based on code by Maziar Raissi

% fits a Hidden Physics Model (Raissi and Karniadakis) to observations of 
% a KS system, with a base SM kernel then integrates the system in time

function KS_integrator_sm()
%% NOTE:

% settings used to generate plots:
% N0 = N1 = 300, nmix = 4
% model.train(500, 100) for both noisy and clean data - this takes a long
% time!
% rng('default')

%% Setup
clc; close all;

addpath ..
addpath ../Utilities
addpath ../Kernels/KS
addpath ../Utilities/export_fig

function CleanupFun()
    rmpath ..
    rmpath ../Utilities
    rmpath ../Kernels/KS
    rmpath ../Utilities/export_fig
end

finishup = onCleanup(@() CleanupFun());

set(0,'defaulttextinterpreter','latex')

%% Load Data, Initialize
load('../Data/kuramoto_sivashinsky.mat','uu','tt','x')
u_star = uu; %512x201
t_star = tt'; %201x1
x_star = x; %512x1
x_star_scl = x_star;%(x-min(x))/(max(x)-min(x));
N_star = size(x_star,1);
nsteps = size(t_star,1)-1;

% number of sample points for the timesteps sampled
N0 = 300;
N1 = 300;

nmix = 6; % number of mixture components in base SM kernel
ncoeffs = 3; % KS has 3 coefficients to learn

% run both clean and noisy simulation?
clean = true;
noisy = true;

rng('default'); % for consistency: other rng settings will give different results
i=randi(nsteps);
i=63; % in order to start before the onset of chaos in the system's dynamics
dt = t_star(i+1)-t_star(i);

% sample from simulation data as ground truth
idx0 = randsample(N_star, N0);
x0 = x_star_scl(idx0,:);
u0 = u_star(idx0,i);

idx1 = randsample(N_star, N1);
x1 = x_star_scl(idx1,:);
u1 = u_star(idx1,i+1);

%% Clean Data Time Extrapolation
if clean
    % hyperparameters are log(noise), and coefficients. SM kernel
    % hyperparameters are set based on U1's spectrum (see
    % SM_HPM_extrap.m).
    hyp = [0 rand(1,ncoeffs)];
    model = SM_HPM_extrap(x1,u1,x0,u0,dt,hyp,nmix,ncoeffs);
    model = model.train(500, 100);

    simdat = u_star(:,(i+2):end);
    tdat = t_star((i+2):end);
    ext_length = size(t_star, 1) - (i+2);

    [unext, varnext] = model.extrap_predict(x1, u1, x_star_scl);
    preds = zeros(size(x_star_scl, 1),ext_length); % predictive means
    vars = zeros(size(x_star_scl, 1),ext_length); % predictive variances
    preds(:, 1) = unext;
    vars(:, 1) = varnext;

    for k=2:(nsteps-i)
        [temp1, temp2] = model.extrap_predict(x_star_scl, preds(:,k-1), x_star_scl);
        preds(:, k) = temp1;
        vars(:, k) = temp2;
    end

    % Save results in .mat file to be plotted as vector graphics later
    % Will be saved after running the noisy simulation if noisy is set to true.
    results.x = x_star;
    results.t = tdat;
    results.cleanpreds = preds;
    results.cleanvars = vars;
    results.cleanhyps = model.hyp;
    results.simdat = simdat;
    if ~noisy
        save('../Results/smKScleanresults.mat','-struct','results');
    end

    % visualize results
    fig = figure('Name','Clean Extrapolation Results','NumberTitle','off');
    hold on
    subplot(8,2,1:4)
    plot_surface(tdat, x_star, preds, '$t$', '$x$', 'GP Extrapolation Results');
    subplot(8,2,7:10)
    plot_surface(tdat, x_star, simdat, '$t$', '$x$', 'Simulation Results');
    subplot(8,2,[13 15])
    plot_surface(tdat, x_star, abs(preds-simdat), '$t$', '$x$', 'Total Error Magnitude');
    subplot(8,2,[14 16])
    plot_surface(tdat, x_star, abs(vars), '$t$', '$x$', 'Variance Magnitude');
    set(fig, 'units', 'centimeters', 'position', [1 1 31 21]);
    hold off
    
    params = model.hyp(2:end);
    
    [pred_n_star, var_n_star] = model.predict(x_star_scl);
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
    
    drawnow;
    if noisy
        input('Press Enter to Continue');
    end
end

%% Time Extrapolation With 1% Noise
if noisy
    noise = 0.01;
    rng('default');
    % same sample points, now with 1% noise in measurement
    u0 = u0 + noise*std(u0)*randn(size(u0));
    u1 = u1 + noise*std(u1)*randn(size(u1));

    hyp = [0 rand(1,ncoeffs)];
    model = SM_HPM_extrap(x1,u1,x0,u0,dt,hyp,nmix,ncoeffs);
    model = model.train(500, 100);

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
        results.noisyhyps = model.hyp;
        results.simdat = simdat;
        save('../Results/smKSnoisyresults.mat','-struct','results');
    else
        results.noisypreds = preds;
        results.noisyvars = vars;
        results.noisyhyps = model.hyp;
        save('../Results/smKSresults.mat','-struct','results');
    end

    % visualize results
    fig = figure('Name','Noisy Extrapolation Results','NumberTitle','off');
    hold on
    subplot(8,2,1:4)
    plot_surface(tdat, x_star, preds, '$t$', '$x$', 'GP Extrapolation Results');
    subplot(8,2,7:10)
    plot_surface(tdat, x_star, simdat, '$t$', '$x$', 'Simulation Results');
    subplot(8,2,[13 15])
    plot_surface(tdat, x_star, abs(preds-simdat), '$t$', '$x$', 'Error Magnitude');
    subplot(8,2,[14 16])
    plot_surface(tdat, x_star, abs(vars), '$t$', '$x$', 'Variance Magnitude');
    set(fig, 'units', 'centimeters', 'position', [1 1 31 21]);
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
end