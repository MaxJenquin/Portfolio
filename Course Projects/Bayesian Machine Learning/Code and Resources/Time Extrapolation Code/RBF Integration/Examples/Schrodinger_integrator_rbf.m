% @author: Maxwell Jenquin

function Schrodinger_integrator_rbf()
%% Preliminaries
clc; close all;

addpath ..
addpath ../Utilities
addpath ../Kernels/Schrodinger
addpath ../Utilities/export_fig

function CleanupFun()
    rmpath ..
    rmpath ../Utilities
    rmpath ../Kernels/Schrodinger
    rmpath ../Utilities/export_fig
end

finishup = onCleanup(@() CleanupFun());

set(0,'defaulttextinterpreter','latex')

%% Load Data, Initialize
load('../Data/nls.mat', 'usol', 't', 'x')
u_star = real(usol)'; % 501x512
v_star = imag(usol)'; % 501x512
t_star = t; % 501x1
x_star = x';   % 512x1
N_star = size(x_star,1);
nsteps = size(t_star,1)-1;

N0 = 111;
N1 = 109;

% run both clean and noisy simulation?
clean = true;
noisy = true;

% choose sample time
s=150;
dt = t_star(s+1)-t_star(s);
% choose spatial sample points
idx0 = randsample(N_star, N0);
x0 = x_star(idx0,:);
U0 = [u_star(idx0,s) v_star(idx0,s)];
idx1 = randsample(N_star,N1);
x1 = x_star(idx1,:);
U1 = [u_star(idx1,s+1) v_star(idx1,s+1)];

hyp = [log([1.0 1.0]) log([1.0 1.0]) 0.0 0.0 -4.0];

simdat = u_star(:,(s+2):end);
tdat = t_star((s+2):end);
ext_length = size(t_star, 1) - (s+2);

%% Clean Data Time Extrapolation
if clean
    % initialize and train model
    model = RBF_HPM_extrap(x1, U1, x0, U0, dt, hyp);
    model = model.train(5000);
    
    % extrapolate in time until simulation data ends
    % first extrapolation
    [unext, varnext] = model.extrap_predict(x1, U1, x_star);
    preds = zeros(2*size(x_star, 1),ext_length); % predictive means
    vars = zeros(2*size(x_star, 1),ext_length); % predictive variances
    preds(:, 1) = unext;
    vars(:, 1) = varnext;
    % subsequent extrapolations
    for k=2:(nsteps-s)
        [temp1, temp2] = model.extrap_predict(x_star, reshape(preds(:,k-1),[],2), x_star);
        preds(:, k) = temp1;
        vars(:, k) = temp2;
    end
    
    % Save results in .mat file to be plotted as vector graphics later
    % Will be saved after running the noisy simulation if noisy is set to true.
    u_preds = preds(1:size(x_star,1),:);
    u_vars = vars(1:size(x_star,1),:);
    
    results.x = x_star;
    results.t = tdat;
    results.cleanpreds = u_preds;
    results.cleanvars = u_vars;
    results.cleanhyps = model.hyp;
    results.simdat = simdat; % note this simdat only contains comparison for u_preds
    if ~noisy
        save('../Results/rbfNLScleanresults.mat','-struct','results');
    end
    
    % visualize results
    fig = figure('Name','Clean Extrapolation Results','NumberTitle','off');
    hold on
    subplot(8,2,1:4)
    plot_surface(tdat, x_star, u_preds, '$t$', '$x$', 'GP Extrapolation Results');
    subplot(8,2,7:10)
    plot_surface(tdat, x_star, simdat, '$t$', '$x$', 'Simulation Results');
    subplot(8,2,[13 15])
    plot_surface(tdat, x_star, abs(u_preds-simdat), '$t$', '$x$', 'Total Error Magnitude');
    subplot(8,2,[14 16])
    plot_surface(tdat, x_star, abs(u_vars), '$t$', '$x$', 'Variance Magnitude');
    set(fig, 'units', 'centimeters', 'position', [1 1 31 21]);
    hold off
    
    params = model.hyp;

    [pred_n_star, var_n_star] = model.predict(x_star);
    pred_U_star = reshape(pred_n_star,N_star,2);
    var_n_star = abs(diag(var_n_star));
    var_U_star = reshape(var_n_star,N_star,2);

    error_u = norm(pred_U_star(:,1) - u_star(:,s+1))/norm(u_star(:,s+1));
    error_v = norm(pred_U_star(:,2) - v_star(:,s+1))/norm(v_star(:,s+1));

    fprintf(1,'=========================\n');
    fprintf(1,'Step: %d, Time = %.2f\n\nNLML = %.2f, Error_u = %.2e, Error_v = %.2e\n\n', s, ...
        t_star(s+1), model.NLML, error_u, error_v);

    str = sprintf('%.4f  ', params);
    fprintf('Parameters: %s\n\n', str)
    fprintf(1,'=========================\n\n');


    figure();
    subplot(2,1,1)
    plot_prediction_1D(x_star, u_star(:,s+1), pred_U_star(:,1), var_U_star(:,1), ...
        '$x$', '$u(t,x)$', 'Prediction (clean data)');
    
    subplot(2,1,2)
    plot_prediction_1D(x_star, v_star(:,s+1), pred_U_star(:,2), var_U_star(:,2), ...
        '$x$', '$v(t,x)$', 'Prediction (clean data)');
    
    drawnow;
    if noisy
        input('Press Enter to Continue');
    end
end

%% Time Extrapolation With 1% Noise

if noisy
    % add noise
    noise = 0.01;
    U0(:,1) = U0(:,1) + noise*std(U0(:,1))*randn(size(U0(:,1)));
    U0(:,2) = U0(:,2) + noise*std(U0(:,2))*randn(size(U0(:,2)));
    U1(:,1) = U1(:,1) + noise*std(U1(:,1))*randn(size(U1(:,1)));
    U1(:,2) = U1(:,2) + noise*std(U1(:,2))*randn(size(U1(:,2)));
    % initialize and train model
    model = RBF_HPM_extrap(x1, U1, x0, U0, dt, hyp);
    model = model.train(5000);
    
    % extrapolate in time until simulation data ends
    % first extrapolation
    [unext, varnext] = model.extrap_predict(x1, U1, x_star);
    preds = zeros(2*size(x_star, 1),ext_length); % predictive means
    vars = zeros(2*size(x_star, 1),ext_length); % predictive variances
    preds(:, 1) = unext;
    vars(:, 1) = varnext;
    % subsequent extrapolations
    for k=2:(nsteps-s)
        [temp1, temp2] = model.extrap_predict(x_star, reshape(preds(:,k-1),[],2), x_star);
        preds(:, k) = temp1;
        vars(:, k) = temp2;
    end
    
    u_preds = preds(1:size(x_star,1),:);
    u_vars = vars(1:size(x_star,1),:);
    
    if ~clean
        results.x = x_star;
        results.t = tdat;
        results.noisypreds = u_preds;
        results.noisyvars = u_vars;
        results.noisyhyps = model.hyp;
        results.simdat = simdat;
        save('../Results/rbfNLSnoisyresults.mat','-struct','results');
    else
        results.noisypreds = u_preds;
        results.noisyvars = u_vars;
        results.noisyhyps = model.hyp;
        save('../Results/rbfNLSresults.mat','-struct','results');
    end
    % visualize results
    fig = figure('Name','Noisy Extrapolation Results','NumberTitle','off');
    hold on
    subplot(8,2,1:4)
    plot_surface(tdat, x_star, u_preds, '$t$', '$x$', 'GP Extrapolation Results (real part)');
    subplot(8,2,7:10)
    plot_surface(tdat, x_star, simdat, '$t$', '$x$', 'Simulation Results');
    subplot(8,2,[13 15])
    plot_surface(tdat, x_star, abs(u_preds-simdat), '$t$', '$x$', 'Error Magnitude');
    subplot(8,2,[14 16])
    plot_surface(tdat, x_star, abs(u_vars), '$t$', '$x$', 'Variance Magnitude');
    set(fig, 'units', 'centimeters', 'position', [1 1 31 21]);
    hold off

    params_noise = model.hyp;

    [pred_n_star, var_n_star] = model.predict(x_star);
    pred_U_star = reshape(pred_n_star,N_star,2);
    var_n_star = abs(diag(var_n_star));
    var_U_star = reshape(var_n_star,N_star,2);

    error_u = norm(pred_U_star(:,1) - u_star(:,s+1))/norm(u_star(:,s+1));
    error_v = norm(pred_U_star(:,2) - v_star(:,s+1))/norm(v_star(:,s+1));

    fprintf(1,'=========================\n');
    fprintf(1,'Step: %d, Time = %.2f\n\nNLML = %.2f, Error_u = %.2e, Error_v = %.2e\n\n', s, ...
        t_star(s+1), model.NLML, error_u, error_v);

    str = sprintf('%.4f  ', params_noise);
    fprintf('Parameters: %s\n\n', str)
    fprintf(1,'=========================\n\n');
    
    figure();
    subplot(2,1,1)
    plot_prediction_1D(x_star, u_star(:,s+1), pred_U_star(:,1), var_U_star(:,1), ...
        '$x$', '$u(t,x)$', 'Prediction (clean data)');
    
    subplot(2,1,2)
    plot_prediction_1D(x_star, v_star(:,s+1), pred_U_star(:,2), var_U_star(:,2), ...
        '$x$', '$v(t,x)$', 'Prediction (clean data)');
    
    drawnow;
end
end