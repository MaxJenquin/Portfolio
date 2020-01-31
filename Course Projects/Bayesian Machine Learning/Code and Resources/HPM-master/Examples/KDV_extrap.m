% @author: Maziar Raissi

% modified from KDV in same directory by Maxwell Jenquin

function KDV_extrap()
clc; close all;

plt = 1;
plt_pred = 0;
save_plt = 1;

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
nsteps = size(t_star,1);
    
%% Setup
N0 = 111;
N1 = 109;
%% Clean Data
rng('default')
i = 70;
dt = t_star(i+1) - t_star(i);

idx0 = randsample(N_star, N0);
x0 = x_star(idx0,:);
u0 = u_star(idx0,i);

idx1 = randsample(N_star,N1);
x1 = x_star(idx1,:);
u1 = u_star(idx1,i+1);

hyp = [log([1.0 1.0]) 0.0 0.0 -4.0];
model = HPM_extrap_old(x1, u1, x0, u0, dt, hyp);
model = model.train(5000);

[unext, varnext] = model.extrap_predict(x1, u1, x_star);
preds=[unext];
vars=[varnext];
for k=2:(nsteps-i)
    [temp1, temp2] = model.extrap_predict(x_star, preds(:,k-1), x_star); 
    preds=[preds, temp1];
    vars=[vars, temp2];
end

figure
ylim([-max(unext)-1,max(unext)+1])
for k=1:(nsteps-i)
    plot(x_star,preds(:,k),'k')
    pause(.01);
end

fig=figure();
set(fig,'units','normalized','outerposition',[0 0 1 .5])
subplot(3,2,1:2)
plot_surface(t_star, x_star, u_star, '$t$', '$x$', 'Simulated $u(t,x)$');

subplot(3,2,3:4)
plot_surface(t_star, x_star, preds, '$t$', '$x$', 'Approximated $u(t,x)$');

subplot(3,2,5:6)
plot_surface(t_star, x_star, abs(preds-u_star), '$t$', '$x$', 'Error');
    
pause();

hyp = model.hyp;
params = hyp(3:4);

[pred_n_star, var_n_star] = model.predict(x_star);
var_n_star = abs(diag(var_n_star));

error = norm(pred_n_star - u_star(:,i+1))/norm(u_star(:,i+1));

fprintf(1,'=========================\n');
fprintf(1,'Step: %d, Time = %.2f\n\nNLML = %.2f, Error = %.2e\n\n', i, ...
    t_star(i+1), model.NLML, error);

str = sprintf('%.4f  ', params);
fprintf('Parameters: %s\n\n', str)
fprintf(1,'=========================\n\n');

if plt_pred == 1
    figure();
    plot_prediction_1D(x_star, u_star(:,i+1), pred_n_star, var_n_star, ...
        '$x$', '$u(t,x)$', 'Prediction (clean data)');
    
    drawnow;
end

%% Noisy Data
noise = 0.01;
u0 = u0 + noise*std(u0)*randn(size(u0));
u1 = u1 + noise*std(u1)*randn(size(u1));

hyp = [log([1.0 1.0]) 0.0 0.0 -4.0];
model = HPM_extrap_old(x1, u1, x0, u0, dt, hyp);
model = model.train(5000);

hyp = model.hyp;
params_noise = hyp(3:4);

[pred_n_star, var_n_star] = model.predict(x_star);
var_n_star = abs(diag(var_n_star));

error = norm(pred_n_star - u_star(:,i+1))/norm(u_star(:,i+1));

fprintf(1,'=========================\n');
fprintf(1,'Step: %d, Time = %.2f\n\nNLML = %.2f, Error = %.2e\n\n', i, ...
    t_star(i+1), model.NLML, error);

str = sprintf('%.4f  ', params_noise);
fprintf('Parameters: %s\n\n', str)
fprintf(1,'=========================\n\n');

if plt_pred == 1
    figure();
    plot_prediction_1D(x_star, u_star(:,i+1), pred_n_star, var_n_star, ...
        '$x$', '$u(t,x)$', 'Prediction (noisy data)');
    
    drawnow;
end

%% Plot Results

if plt == 1
    fig = figure();
    set(fig,'units','normalized','outerposition',[0 0 1 .5])
    subplot(3,2,1:2)
    plot_surface(t_star, x_star, u_star, '$t$', '$x$', '$u(t,x)$');
    
    hold on
    plot3([t_star(i) t_star(i)],get(gca,'ylim'),[10 10],'w','LineWidth',2)
    plot3([t_star(i+1) t_star(i+1)],get(gca,'ylim'),[10 10],'w','LineWidth',2)
    
    subplot(3,2,3);
    tit = sprintf('$t = $ %.2f\n%d training data\n', t_star(i), N0);
    plot_data_1D(x_star, u_star(:,i), x0, u0, '$x$', '$u(t,x)$', tit);
    
    subplot(3,2,4);
    tit = sprintf('$t = $ %.2f\n%d training data\n', t_star(i+1), N1);
    plot_data_1D(x_star, u_star(:,i+1), x1, u1, '$x$', '$u(t,x)$', tit);
    
    subplot(3,2,5:6);
    s1 = '$\begin{tabular}{ |c|c| }  \hline Correct PDE & $u_t + 6 u u_x + u_{xxx} = 0$ \\  \hline Identified PDE (clean data) &';
    s2 = sprintf('$u_t + %.3f u u_x + %.3f u_{xxx} = 0$', params(1), params(2));
    s3 = ' \\  \hline Identified PDE (1\% noise) &';
    s4 = sprintf('$u_t + %.3f u u_x + %.3f u_{xxx} = 0$', params_noise(1), params_noise(2));
    s5 = ' \\  \hline \end{tabular}$';
    s = strcat(s1,s2,s3,s4,s5);
    text(0.1,0.8,s,'interpreter','latex','FontSize',18)
    axis off
    
    if save_plt == 1
        export_fig ../Figures/KDV.png -r300
    end
    
    drawnow();
end

end