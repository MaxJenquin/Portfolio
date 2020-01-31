% @author: Maxwell Jenquin

function [diffs, times, ndiffs, ntimes] = Schrodinger_Sample_Test()

%% Preliminaries
clc; close all;

clean = false;
noisy = true;

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

diffs = 0;
times = 0;
ndiffs = 0;
ntimes = 0;

%% Load Data
load('../Data/nls.mat', 'usol', 't', 'x')
u_star = real(usol)'; % 501x512
v_star = imag(usol)'; % 501x512
t_star = t; % 501x1
x_star = x';   % 512x1
N_star = size(x_star,1);
nsteps = size(t_star,1)-1;
N0 = 49; % number of samples at first slice
N1 = 51; % number of samples at second slice

%% Noise Free Testing
% test the fit of the model based on where the sample slices occur
if clean
    u0samplepoints = randsample(N_star, N0);
    u1samplepoints = randsample(N_star, N1);
    x0 = x_star(u0samplepoints, :);
    x1 = x_star(u1samplepoints, :);

    coeffs = zeros(2, nsteps);
    times = zeros(1, nsteps);
    for k=1:nsteps
        disp(k)
        %timestep length
        dt = t_star(k+1)-t_star(k);
        %sample from simulation data, noise-free
        U0 = [u_star(u0samplepoints,k) v_star(u0samplepoints,k)];
        U1 = [u_star(u1samplepoints,k+1) v_star(u1samplepoints,k+1)];
        % initialize hyperparameters and model, suppressing text output
        hyp = [log([1.0 1.0]) log([1.0 1.0]) 0.0 0.0 -4.0];
        [TextDump, model] = evalc('HPM(x1, U1, x0, U0, dt, hyp);');
        % train model, suppressing text output
        tstart=tic;
        [TextDump, model] = evalc('model.train(5000);');
        times(k)=toc(tstart);
        % get PDE coefficients
        hypers = model.hyp;
        coeffs(1, k) = hypers(5); % coefficient on h_{xx} term
        coeffs(2, k) = hypers(6); % coefficient on h*|h|^2 term
    end
    truecoeffs = [(0.5)*ones(1,nsteps); ones(1,nsteps)];
    diffs = (truecoeffs-coeffs)./truecoeffs;
    disp(mean(times));
    disp(std(times));

    %get the sample point Raissi and Karniadakis used
    rng('default')
    s = randi(nsteps);

    %plot results
    fig=figure('Name','Clean Results','NumberTitle','off');
    hold on
    plot(diffs(1,:),'LineWidth',3)
    plot(diffs(2,:),'LineWidth',3)
    line([s,s],ylim,'Color','g','LineStyle','--');
    hline=refline([0,0]);
    hline.Color='k';
    title('Relative Coefficient Error vs Sampled Timesteps, NLS','FontWeight','Bold')
    xlabel('Timestep Sampled')
    ylabel('Estimated Coefficient Error Ratio')
    set(fig, 'units', 'centimeters', 'position', [1 1 17 13])
    legend({'Coefficient 1', 'Coefficient 2', 'Sampled Timestep'},'Location','northwest');
end
%% Noisy Testing
if noisy
    u0samplepoints = randsample(N_star, N0);
    u1samplepoints = randsample(N_star, N1);
    x0 = x_star(u0samplepoints, :);
    x1 = x_star(u1samplepoints, :);

    coeffs = zeros(2, nsteps);
    ntimes = zeros(1, nsteps);
    for k=1:nsteps
        disp(k)
        %timestep length
        dt = t_star(k+1)-t_star(k);
        %sample from simulation data, with 1% noise
        U0 = [u_star(u0samplepoints,k) v_star(u0samplepoints,k)];
        U1 = [u_star(u1samplepoints,k+1) v_star(u1samplepoints,k+1)];
        noise = 0.01;
        U0(:,1) = U0(:,1) + noise*std(U0(:,1))*randn(size(U0(:,1)));
        U0(:,2) = U0(:,2) + noise*std(U0(:,2))*randn(size(U0(:,2)));
        U1(:,1) = U1(:,1) + noise*std(U1(:,1))*randn(size(U1(:,1)));
        U1(:,2) = U1(:,2) + noise*std(U1(:,2))*randn(size(U1(:,2)));
        % initialize hyperparameters and model, suppressing text output
        hyp = [log([1.0 1.0]) log([1.0 1.0]) 0.0 0.0 -4.0];
        [TextDump, model] = evalc('HPM(x1, U1, x0, U0, dt, hyp);');
        % train model, suppressing text output
        tstart=tic;
        [TextDump, model] = evalc('model.train(5000);');
        ntimes(k)=toc(tstart);
        % get PDE coefficients
        hypers = model.hyp;
        coeffs(1, k) = hypers(5); % coefficient on h_{xx} term
        coeffs(2, k) = hypers(6); % coefficient on h*|h|^2 term
    end
    truecoeffs = [(0.5)*ones(1,nsteps); ones(1,nsteps)];
    ndiffs = (truecoeffs-coeffs)./truecoeffs;
    disp(mean(ntimes));
    disp(std(ntimes));

    %get the sample point Raissi and Karniadakis used
    rng('default')
    s = randi(nsteps);

    %plot results
    fig=figure('Name','Noisy Results','NumberTitle','off');
    hold on
    plot(ndiffs(1,:),'LineWidth',3)
    plot(ndiffs(2,:),'LineWidth',3)
    line([s,s],ylim,'Color','g','LineStyle','--');
    hline=refline([0,0]);
    hline.Color='k';
    title('Relative Coefficient Error vs Sampled Timesteps, NLS','FontWeight','Bold')
    xlabel('Timestep Sampled')
    ylabel('Estimated Coefficient Error Ratio')
    set(fig, 'units', 'centimeters', 'position', [1 1 17 13])
    legend({'Coefficient 1', 'Coefficient 2', 'Sampled Timestep'},'Location','northwest');
end
end