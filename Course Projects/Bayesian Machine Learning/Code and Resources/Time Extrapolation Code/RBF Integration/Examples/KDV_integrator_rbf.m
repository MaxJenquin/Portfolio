% @author: Maxwell Jenquin, based on code by Maziar Raissi

% fits a Hidden Physics Model (Raissi and Karniadakis) to observations of a
% KDV system, with a base RBF kernel then integrates the system in time

function KDV_integrator_rbf()
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
load('../Data/kdv.mat', 'usol', 't', 'x')
u_star = real(usol); % 512x201
t_star = t; % 201x1
x_star = x';   % 512x1
N_star = size(x_star,1);
nsteps = size(t_star,1)-1;

N0 = 111;
N1 = 109;

% run both clean and noisy simulations?
clean = true;
noisy = true;

% choose sample time
rng('default')
s=randi(nsteps);
dt = t_star(s+1)-t_star(s);
% choose spatial sample points
idx0 = randsample(N_star, N0);
x0 = x_star(idx0,:);
u0 = u_star(idx0,s);
idx1 = randsample(N_star,N1);
x1 = x_star(idx1,:);
u1 = u_star(idx1,s+1);

% initialize hyperparameters following Maziar and Raissi
hyp = [log([1.0 1.0]) 0.0 0.0 -4.0];

simdat = u_star(:,(s+2):end);
tdat = t_star((s+2):end);
ext_length = size(t_star, 1) - (s+2);

%% Clean Data Time Extrapolation
if clean
    % initialize and train model
    model = RBF_HPM_extrap(x1, u1, x0, u0, dt, hyp);
    model = model.train(5000);
    
    % extrapolate in time until simulation data ends
    % first extrapolation
    [unext, varnext] = model.extrap_predict(x1, u1, x_star);
    preds = zeros(size(x_star, 1),ext_length); % predictive means
    vars = zeros(size(x_star, 1),ext_length); % predictive variances
    preds(:, 1) = unext;
    vars(:, 1) = varnext;
    % subsequent extrapolations
    for k=2:(nsteps-s)
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
    results.cleanhyps = model.hyp;
    results.simdat = simdat;
    if ~noisy
        save('../Results/rbfKDVcleanresults.mat','-struct','results');
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
    
    params = model.hyp;

    [pred_n_star, var_n_star] = model.predict(x_star);
    var_n_star = abs(diag(var_n_star));
    
    error = norm(pred_n_star - u_star(:,s+1))/norm(u_star(:,s+1));
    
    fprintf(1,'=========================\n');
    fprintf(1,'Step: %d, Time = %.2f\n\nNLML = %.2f, Error = %.2e\n\n', s, ...
        t_star(s+1), model.NLML, error);
    
    str = sprintf('%.4f  ', params);
    fprintf('Parameters: %s\n\n', str)
    fprintf(1,'=========================\n\n');
    
    
    figure();
    plot_prediction_1D(x_star, u_star(:,s+1), pred_n_star, var_n_star, ...
        '$x$', '$u(t,x)$', 'Full U1 Prediction (clean data)');
    
    drawnow;
    if noisy
        input('Press Enter to Continue');
    end
end

%% Time Extrapolation With 1% Noise

if noisy
    % add noise
    noise = 0.01;
    u0 = u0 + noise*std(u0)*randn(size(u0));
    u1 = u1 + noise*std(u1)*randn(size(u1));
    % initialize and train model
    model = RBF_HPM_extrap(x1, u1, x0, u0, dt, hyp);
    model = model.train(5000);
    
    % extrapolate in time until simulation data ends
    % first extrapolation
    [unext, varnext] = model.extrap_predict(x1, u1, x_star);
    preds = zeros(size(x_star, 1),ext_length); % predictive means
    vars = zeros(size(x_star, 1),ext_length); % predictive variances
    preds(:, 1) = unext;
    vars(:, 1) = varnext;
    % subsequent extrapolations
    for k=2:(nsteps-s)
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
        save('../Results/rbfKDVnoisyresults.mat','-struct','results');
    else
        results.noisypreds = preds;
        results.noisyvars = vars;
        results.noisyhyps = model.hyp;
        save('../Results/rbfKDVresults.mat','-struct','results');
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

    params_noise = model.hyp;

    [pred_n_star, var_n_star] = model.predict(x_star);
    var_n_star = abs(diag(var_n_star));

    error = norm(pred_n_star - u_star(:,s+1))/norm(u_star(:,s+1));

    fprintf(1,'=========================\n');
    fprintf(1,'Step: %d, Time = %.2f\n\nNLML = %.2f, Error = %.2e\n\n', s, ...
        t_star(s+1), model.NLML, error);

    str = sprintf('%.4f  ', params_noise);
    fprintf('Parameters: %s\n\n', str)
    fprintf(1,'=========================\n\n');

    figure();
    plot_prediction_1D(x_star, u_star(:,s+1), pred_n_star, var_n_star, ...
        '$x$', '$u(t,x)$', 'Full U1 Prediction (noisy data)');

    drawnow;
end

end