% @author: Maxwell Jenquin

function KDV_integrator_rbf()
%% Preliminaries
clc; close all;

clean = true;
noisy = false;

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

%% Load Data
load('../Data/kdv.mat', 'usol', 't', 'x')
u_star = real(usol); % 512x201
t_star = t; % 201x1
x_star = x';   % 512x1
N_star = size(x_star,1);
nsteps = size(t_star,1)-1;
N0 = 111;
N1 = 109;

%% Setup
% choose sample time
s=27;
dt = t_star(s+1)-t_star(s);
% choose spatial sample points
idx0 = randsample(N_star, N0);
x0 = x_star(idx0,:);
u0 = u_star(idx0,s);
idx1 = randsample(N_star,N1);
x1 = x_star(idx1,:);
u1 = u_star(idx1,s+1);

hyp = [log([1.0 1.0]) 0.0 0.0 -4.0];

simdat = u_star(:,(s+2):end);
tdat = t_star((s+2):end);
ext_length = size(t_star, 1) - (s+2);

%% Extrapolate PDE Dynamics after fitting to clean data
if clean
    % initialize and train model
    model = HPM_extrap_old(x1, u1, x0, u0, dt, hyp);
    model = model.train(5000);
    
    % extrapolate in time until simulation data ends
    % first extrapolation
    
    sigma = (model.hyp(end))*ones(size(x1,1),1);
    %with HPM_extrap
    %[unext, varnext] = model.extrap_predict(x1, u1, sigma, x_star);
    %with HPM_extrap_old
    [unext, varnext] = model.extrap_predict(x1, u1, x_star);
    preds = zeros(size(x_star, 1),ext_length); % predictive means
    vars = zeros(size(x_star, 1),ext_length); % predictive variances
    preds(:, 1) = unext;
    vars(:, 1) = varnext;
    % subsequent extrapolations
    for k=2:(nsteps-s)
        %with HPM_extrap
        %[temp1, temp2] = model.extrap_predict(x_star, preds(:,k-1), vars(:,k-1), x_star);
        %with HPM_extrap_old
        [temp1, temp2] = model.extrap_predict(x_star, preds(:,k-1), x_star);
        preds(:, k) = temp1;
        vars(:, k) = temp2;
    end
    % plot results
    fig = figure('Name','Clean Extrapolation Results','NumberTitle','off');
    hold on
    subplot(2,1,2)
    plot_surface(tdat, x_star, simdat, '$t$', '$x$', 'Simulation Results');
    subplot(2,1,1)
    plot_surface(tdat, x_star, preds, '$t$', '$x$', 'GP Extrapolation Results');
    %subplot(2,2,3)
    %plot_surface(tdat, x_star, abs(preds-simdat), '$t$', '$x$', 'Error Magnitude');
    %subplot(2,2,4)
    %plot_surface(tdat, x_star, abs(vars), '$t$', '$x$', 'Variance Magnitude');
    %title('KDV Clean Extrapolation vs Simulation','FontWeight','bold');
    set(fig, 'units', 'centimeters', 'position', [1 1 21 11]);
    % make .gif of results
    %top = preds+abs(vars);
    %bot = preds-abs(vars);
    MA=max(preds(:));
    MI=min(preds(:));
    filename = 'KDVclean.gif';
    h = figure('Name','GIF of CLEAN');
    for k=1:ext_length
        plot(preds(:,k),'LineWidth',3,'Color','k');
        %hold on
        %plot(top(:,k),'LineStyle','--','Color','b');
        %plot(bot(:,k),'LineStyle','--','Color','b');
        %hold off
        ylim([MI,MA]);
        xlabel('$x$');
        ylabel ('$u$')
        title('Kortweg-de Vries System Behavior in Expectation');
        drawnow
        
        frame=getframe(h);
        im=frame2im(frame);
        [imind,cm]=rgb2ind(im,256);
        if k==1
            imwrite(imind,cm,filename,'gif','DelayTime',0, 'Loopcount',inf);
        else
            imwrite(imind,cm,filename,'gif','DelayTime',0,'WriteMode','append');
        end
    end
end

%% Extrapolate PDE dynamics after fitting to noisy data

if noisy
    % add noise
    noise = 0.01;
    u0 = u0 + noise*std(u0)*randn(size(u0));
    u1 = u1 + noise*std(u1)*randn(size(u1));
    % initialize and train model
    model = HPM_extrap_old(x1, u1, x0, u0, dt, hyp);
    model = model.train(5000);
    
    % extrapolate in time until simulation data ends
    % first extrapolation
    
    sigma = (model.hyp(end))*ones(size(x1,1),1);
    %with HPM_extrap
    %[unext, varnext] = model.extrap_predict(x1, u1, sigma, x_star);
    %with HPM_extrap_old
    [unext, varnext] = model.extrap_predict(x1, u1, x_star);
    preds = zeros(size(x_star, 1),ext_length); % predictive means
    vars = zeros(size(x_star, 1),ext_length); % predictive variances
    preds(:, 1) = unext;
    vars(:, 1) = varnext;
    % subsequent extrapolations
    for k=2:(nsteps-s)
        %with HPM_extrap
        %[temp1, temp2] = model.extrap_predict(x_star, preds(:,k-1), vars(:,k-1), x_star);
        %with HPM_extrap_old
        [temp1, temp2] = model.extrap_predict(x_star, preds(:,k-1), x_star); 
        preds(:, k) = temp1;
        vars(:, k) = temp2;
    end
    % plot results
    fig = figure('Name','Noisy Extrapolation Results','NumberTitle','off');
    hold on
    subplot(2,1,2)
    plot_surface(tdat, x_star, simdat, '$t$', '$x$', 'Simulation Results');
    subplot(2,1,1)
    plot_surface(tdat, x_star, preds, '$t$', '$x$', 'GP Extrapolation Results');
    %subplot(2,2,3)
    %plot_surface(tdat, x_star, abs(preds-simdat), '$t$', '$x$', 'Error Magnitude');
    %subplot(2,2,4)
    %plot_surface(tdat, x_star, abs(vars), '$t$', '$x$', 'Variance Magnitude');
    %title('KDV Noisy Extrapolation vs Simulation','FontWeight','bold');
    set(fig, 'units', 'centimeters', 'position', [1 1 21 11]);
    % make .gif of results
    %top = preds+abs(vars);
    %bot = preds-abs(vars);
    MA=max(preds(:));
    MI=min(preds(:));
    filename = 'KDVnoisy.gif';
    h = figure('Name','GIF of NOISY');
    for k=1:ext_length
        plot(preds(:,k),'LineWidth',3,'Color','k');
        %hold on
        %plot(top(:,k),'LineStyle','--','Color','b');
        %plot(bot(:,k),'LineStyle','--','Color','b');
        %hold off
        ylim([MI,MA]);
        xlabel('$x$');
        ylabel ('$u$')
        title('Kortweg-de Vries System Behavior in Expectation');        
        drawnow
        
        frame=getframe(h);
        im=frame2im(frame);
        [imind,cm]=rgb2ind(im,256);
        if k==1
            imwrite(imind,cm,filename,'gif','DelayTime',0, 'Loopcount',inf);
        else
            imwrite(imind,cm,filename,'gif','DelayTime',0,'WriteMode','append');
        end
    end
end

end