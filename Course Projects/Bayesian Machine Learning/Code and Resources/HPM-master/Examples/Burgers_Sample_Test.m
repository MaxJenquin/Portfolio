% @author: Maxwell Jenquin

function [diffs, times, ndiffs, ntimes] = Burgers_Sample_Test()
%% Preliminaries
clc; close all;

clean = true;
noisy = true;

addpath ..
addpath ../Utilities
addpath ../Kernels/Burgers
addpath ../Utilities/export_fig

function CleanupFun()
    rmpath ..
    rmpath ../Utilities
    addpath ../Kernels/Burgers
    addpath ../Utilities/export_fig
end

finishup = onCleanup(@() CleanupFun());

set(0,'defaulttextinterpreter','latex')

diffs = 0;
times = 0;
ndiffs = 0;
ntimes = 0;

%% Load Data
load('../Data/burgers.mat', 'usol', 't', 'x')
u_star = real(usol); % 256x101
t_star = t; % 101x1
x_star = x';   % 256x1
N_star = size(x_star,1);
nsteps = size(t_star,1)-1;
N0 = 71;
N1 = 69;

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
        u0 = u_star(u0samplepoints, k);
        u1 = u_star(u1samplepoints, k+1);
        %initialize hyperparameters and model, suppressing text output
        hyp = [log([1.0 1.0]) 0.0 0.0 -4.0];
        [TextDump, model] = evalc('HPM(x1, u1, x0, u0, dt, hyp);');
        % train model, suppressing text output
        tstart=tic;
        [TextDump, model] = evalc('model.train(5000);');
        times(k)=toc(tstart);
        % get PDE coefficients
        hypers = model.hyp;
        coeffs(1, k) = hypers(3); % coefficient on u*u_x term
        coeffs(2, k) = hypers(4); % coefficient on u_{xx} term
    end
    truecoeffs = [ones(1,nsteps); (0.1)*ones(1,nsteps)];
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
    title('Relative Coefficient Error vs Sampled Timesteps, Burgers','FontWeight','Bold')
    xlabel('Timestep Sampled')
    ylabel('Estimated Coefficient Error Ratio')
    set(fig, 'units', 'centimeters', 'position', [1 1 17 13])
    legend('Coefficient 1', 'Coefficient 2', 'Sampled Timestep');
end
%% Noisy Testing
if noisy
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
        %sample from simulation data, with 1% noise
        u0 = u_star(u0samplepoints, k);
        u1 = u_star(u1samplepoints, k+1);
        noise = 0.01;
        u0 = u0 + noise*std(u0)*randn(size(u0));
        u1 = u1 + noise*std(u1)*randn(size(u1));
        %initialize hyperparameters and model, suppressing text output
        hyp = [log([1.0 1.0]) 0.0 0.0 -4.0];
        [TextDump, model] = evalc('HPM(x1, u1, x0, u0, dt, hyp);');
        % train model, suppressing text output
        tstart=tic;
        [TextDump, model] = evalc('model.train(5000);');
        times(k)=toc(tstart);
        % get PDE coefficients
        hypers = model.hyp;
        coeffs(1, k) = hypers(3); % coefficient on u*u_x term
        coeffs(2, k) = hypers(4); % coefficient on u_{xx} term
    end
    truecoeffs = [ones(1,nsteps); (0.1)*ones(1,nsteps)];
    diffs = (truecoeffs-coeffs)./truecoeffs;
    disp(mean(times));
    disp(std(times));

    %get the sample point Raissi and Karniadakis used
    rng('default')
    s = randi(nsteps);

    %plot results
    fig=figure('Name','Noisy Results','NumberTitle','off');
    hold on
    plot(diffs(1,:),'LineWidth',3)
    plot(diffs(2,:),'LineWidth',3)
    line([s,s],ylim,'Color','g','LineStyle','--');
    hline=refline([0,0]);
    hline.Color='k';
    title('Relative Coefficient Error vs Sampled Timesteps, Burgers','FontWeight','Bold')
    xlabel('Timestep Sampled')
    ylabel('Estimated Coefficient Error Ratio')
    set(fig, 'units', 'centimeters', 'position', [1 1 17 13])
    legend('Coefficient 1', 'Coefficient 2', 'Sampled Timestep');
end
end