% @author: Maxwell Jenquin

function Schrodinger_integrator_rbf()
%% Preliminaries
clc; close all;

clean = true;
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

%% Load Data
load('../Data/nls.mat', 'usol', 't', 'x')
u_star = real(usol)'; % 501x512
v_star = imag(usol)'; % 501x512
t_star = t; % 501x1
x_star = x';   % 512x1
N_star = size(x_star,1);
nsteps = size(t_star,1)-1;
N0 = 111;
N1 = 109;

%% Setup
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

%% Extrapolate PDE Dynamics after fitting to clean data
if clean
    % initialize and train model
    model = HPM_extrap_old(x1, U1, x0, U0, dt, hyp);
    model = model.train(5000);
    
    % extrapolate in time until simulation data ends
    % first extrapolation
    
    sigma = (model.hyp(end))*ones(size(x1,1),1);
    %with HPM_extrap
    %[unext, varnext] = model.extrap_predict(x1, u1, sigma, x_star);
    %with HPM_extrap_old
    [unext, varnext] = model.extrap_predict(x1, U1, x_star);
    preds = zeros(2*size(x_star, 1),ext_length); % predictive means
    vars = zeros(2*size(x_star, 1),ext_length); % predictive variances
    preds(:, 1) = unext;
    vars(:, 1) = varnext;
    % subsequent extrapolations
    for k=2:(nsteps-s)
        %with HPM_extrap
        %[temp1, temp2] = model.extrap_predict(x_star, preds(:,k-1), vars(:,k-1), x_star);
        %with HPM_extrap_old
        [temp1, temp2] = model.extrap_predict(x_star, reshape(preds(:,k-1),[],2), x_star);
        preds(:, k) = temp1;
        vars(:, k) = temp2;
    end
    % plot results
    u_preds = preds(1:size(x_star,1),:);
    u_vars = vars(1:size(x_star,1),:);
    fig = figure('Name','Clean Extrapolation Results','NumberTitle','off');
    hold on
    subplot(2,1,2)
    plot_surface(tdat, x_star, simdat, '$t$', '$x$', 'Simulation Results');
    subplot(2,1,1)
    plot_surface(tdat, x_star, u_preds, '$t$', '$x$', 'GP Extrapolation Results');
    %subplot(2,2,3)
    %plot_surface(tdat, x_star, abs(u_preds-simdat), '$t$', '$x$', 'Error Magnitude');
    %subplot(2,2,4)
    %plot_surface(tdat, x_star, abs(u_vars), '$t$', '$x$', 'Variance Magnitude');
    %title('KDV Clean Extrapolation vs Simulation','FontWeight','bold');
    set(fig, 'units', 'centimeters', 'position', [1 1 21 11]);
    hold off
    % make .gif of results
    %top = u_preds+abs(u_vars);
    %bot = u_preds-abs(u_vars);
    MA=max(preds(:));
    MI=min(preds(:));
    filename = 'NLSclean.gif';
    h = figure('Name','GIF of CLEAN');
    for k=1:ext_length
        plot(u_preds(:,k),'LineWidth',3,'Color','k');
        %plot(v_preds(:,k),'LineWidth',3,'Color','b');
        %hold on
        %plot(top(:,k),'LineStyle','--','Color','b');
        %plot(bot(:,k),'LineStyle','--','Color','b');
        %hold off
        xlabel('$x$');
        ylabel ('$u$')
        title('Nonlinear Schr{\"o}dinger System Behavior in Expectation');
        ylim([MI,MA]);
        %legend('Real Part','Imaginary Part')
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
    U0(:,1) = U0(:,1) + noise*std(U0(:,1))*randn(size(U0(:,1)));
    U0(:,2) = U0(:,2) + noise*std(U0(:,2))*randn(size(U0(:,2)));
    U1(:,1) = U1(:,1) + noise*std(U1(:,1))*randn(size(U1(:,1)));
    U1(:,2) = U1(:,2) + noise*std(U1(:,2))*randn(size(U1(:,2)));
    % initialize and train model
    model = HPM_extrap_old(x1, U1, x0, U0, dt, hyp);
    model = model.train(5000);
    
    % extrapolate in time until simulation data ends
    % first extrapolation
    
    sigma = (model.hyp(end))*ones(size(x1,1),1);
        %with HPM_extrap
    %[unext, varnext] = model.extrap_predict(x1, u1, sigma, x_star);
    %with HPM_extrap_old
    [unext, varnext] = model.extrap_predict(x1, U1, x_star);
    preds = zeros(2*size(x_star, 1),ext_length); % predictive means
    vars = zeros(2*size(x_star, 1),ext_length); % predictive variances
    preds(:, 1) = unext;
    vars(:, 1) = varnext;
    % subsequent extrapolations
    for k=2:(nsteps-s)
        %with HPM_extrap
        %[temp1, temp2] = model.extrap_predict(x_star, preds(:,k-1), vars(:,k-1), x_star);
        %with HPM_extrap_old
        [temp1, temp2] = model.extrap_predict(x_star, reshape(preds(:,k-1),[],2), x_star);
        preds(:, k) = temp1;
        vars(:, k) = temp2;
    end
    % plot results
    u_preds = preds(1:size(x_star,1),:);
    u_vars = vars(1:size(x_star,1),:);
    fig = figure('Name','Noisy Extrapolation Results','NumberTitle','off');
    hold on
    subplot(2,1,2)
    plot_surface(tdat, x_star, simdat, '$t$', '$x$', 'Simulation Results');
    subplot(2,1,1)
    plot_surface(tdat, x_star, u_preds, '$t$', '$x$', 'GP Extrapolation Results');
    %subplot(2,2,3)
    %plot_surface(tdat, x_star, abs(u_preds-simdat), '$t$', '$x$', 'Error Magnitude');
    %subplot(2,2,4)
    %plot_surface(tdat, x_star, abs(u_vars), '$t$', '$x$', 'Variance Magnitude');
    %title('KDV Clean Extrapolation vs Simulation','FontWeight','bold');
    set(fig, 'units', 'centimeters', 'position', [1 1 21 11]);
    hold off
    % make .gif of results
    %top = u_preds+abs(u_vars);
    %bot = u_preds-abs(u_vars);
    MA=max(preds(:));
    MI=min(preds(:));
    filename = 'NLSnoisy.gif';
    h = figure('Name','GIF of NOISY');
    for k=1:ext_length
        plot(u_preds(:,k),'LineWidth',3,'Color','k');
        %plot(v_preds(:,k),'LineWidth',3,'Color','b');
        %hold on
        %plot(top(:,k),'LineStyle','--','Color','b');
        %plot(bot(:,k),'LineStyle','--','Color','b');
        %hold off
        xlabel('$x$');
        ylabel ('$u$')
        title('Nonlinear Schr{\"o}dinger System Behavior in Expectation');
        ylim([MI,MA]);
        %legend('Real Part','Imaginary Part')
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