% @author: Maxwell Jenquin

function [diffs, times, ndiffs, ntimes] = KDV_Sample_Test()
%% Preliminaries
clc; close all;

clean = false;
noisy = true;

addpath ..
addpath ../Utilities
addpath ../Kernels/KDV
addpath ../Utilities/export_fig

function CleanupFun()
    rmpath ..
    rmpath ../Utilities
    addpath ../Kernels/KDV
    addpath ../Utilities/export_fig
end

finishup = onCleanup(@() CleanupFun());

set(0,'defaulttextinterpreter','latex')

diffs = 0;
times = 0;
ndiffs = 0;
ntimes = 0;

%% Load Data
load('../Data/kdv.mat', 'usol', 't', 'x')
u_star = real(usol); % 512x201
t_star = t; % 201x1
x_star = x';   % 512x1
N_star = size(x_star,1);
nsteps = size(t_star,1)-1;
N0 = 111; % number of samples at first slice
N1 = 109; % number of samples at second slice

%% Noise Free Testing
% test the fit of the model based on where the sample slices occur
if clean
    rng('default')
    s1 = randi(nsteps)
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
        hyp = [log([1.0,1.0]) 0.0 0.0 -4.0];
        [TextDump, model] = evalc('HPM(x1, u1, x0, u0, dt, hyp);');
        % train model, suppressing text output
        tstart=tic;
        [TextDump, model] = evalc('model.train(5000);');
        times(k)=toc(tstart);
        % get PDE coefficients
        hypers = model.hyp;
        coeffs(1, k) = hypers(3); % coefficient on u*u_x term
        coeffs(2, k) = hypers(4); % coefficient on u_{xxx} term
    end
    truecoeffs = [6*ones(1,nsteps); ones(1,nsteps)];
    diffs = (truecoeffs-coeffs)./truecoeffs;
    disp(mean(times));
    disp(std(times));

    %get the sample point Raissi and Karniadakis used
    

    %plot results
    fig=figure('Name','Clean Results','NumberTitle','off');
    hold on
    plot(diffs(1,:))
    plot(diffs(2,:))
    line([s1,s1],ylim,'Color','g','LineStyle','--');
    hline=refline([0,0]);
    hline.Color='k';
    title('Relative Coefficient Error vs Sampled Timesteps, KDV','FontWeight','Bold')
    xlabel('Timestep Sampled')
    ylabel('Estimated Coefficient Error Ratio')
    set(fig, 'units', 'centimeters', 'position', [1 1 9 7])
    legend('Coefficient 1', 'Coefficient 2', 'Sampled Timestep');
    hold off
end

%% Noisy Data Testing
if noisy
    rng('default')
    s = randi(nsteps)
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
        u0 = u_star(u0samplepoints, k);
        u1 = u_star(u1samplepoints, k+1);
        noise = 0.01;
        u0 = u0 + noise*std(u0)*randn(size(u0));
        u1 = u1 + noise*std(u1)*randn(size(u1));
        %initialize hyperparameters and model, suppressing text output
        hyp = [log([1.0,1.0]) 0.0 0.0 -4.0];
        [TextDump, model] = evalc('HPM(x1, u1, x0, u0, dt, hyp);');
        % train model, suppressing text output
        tstart=tic;
        [TextDump, model] = evalc('model.train(5000);');
        ntimes(k)=toc(tstart);
        % get PDE coefficients
        hypers = model.hyp;
        coeffs(1, k) = hypers(3); % coefficient on u*u_x term
        coeffs(2, k) = hypers(4); % coefficient on u_{xxx} term
    end
    truecoeffs = [6*ones(1,nsteps); ones(1,nsteps)];
    ndiffs = (truecoeffs-coeffs)./truecoeffs;
    disp(mean(ntimes));
    disp(std(ntimes));

    %get the sample point Raissi and Karniadakis used
    rng('default')
    s = randi(nsteps);

    %plot results
    fig=figure('Name','Noisy Results','NumberTitle','off');
    hold on
    plot(ndiffs(1,:))
    plot(ndiffs(2,:))
    line([s,s],ylim,'Color','g','LineStyle','--');
    hline=refline([0,0]);
    hline.Color='k';
    title('Relative Coefficient Error vs Sampled Timesteps, KDV','FontWeight','Bold')
    xlabel('Timestep Sampled')
    ylabel('Estimated Coefficient Error Ratio')
    set(fig, 'units', 'centimeters', 'position', [1 1 9 7])
    legend('Coefficient 1', 'Coefficient 2', 'Sampled Timestep');
    hold off
end
fu = figure();
set(fu, 'units', 'centimeters', 'position', [1 1 19 9])
p = subplot(6,2,11.5);
ps=get(p,'position');
fig = figure();
set(fig, 'units', 'centimeters', 'position', [1 1 19 9])
subplot(6,2,[1 3 5 7])
hold on
plot(diffs(1,:));
plot(diffs(2,:));
line([s1,s1],[-0.1,0.3],'Color','g','LineStyle','--');
hline=refline([0,0]);
hline.Color='k';
xlabel('Timestep')
ylabel('Relative Error')
title('Clean Data')
hold off
subplot(6,2,[2 4 6 8])
hold on
plot(ndiffs(1,:));
plot(ndiffs(2,:))
line([s,s],[-0.4,0.6],'Color','g','LineStyle','--');
hline=refline([0,0]);
hline.Color='k';
xlabel('Timestep')
ylabel('Relative Error')
title('1\% Noise')
hold off
hl=legend('Coefficient 1', 'Coefficient 2', 'Sampled Timestep');
set(hl,'Position',ps);


end
