function KS_gif_maker()
%% Preliminaries
clc; close all;

clean = true;
noisy = true;

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

%% Load Data
load('../Data/kuramoto_sivishinky.mat', 'uu', 'tt', 'x')
u_star = uu; % 1024x251
t_star = tt'; % 251x1
x_star = x;   % 1024x1
N_star = size(x_star,1);
nsteps = size(t_star,1)-1;
    
N0 = 301;
N1 = 299;

%% Setup
% choose sample time
s=61;
ext_length = size(t_star, 1) - (s+2);

MA=max(u_star(:));
MI=min(u_star(:));

filename = 'KSsim.gif';
h=figure();
for k=1:ext_length
    plot(u_star(:,k+s+2),'LineWidth',3,'Color','b');
    hold on
    hold off
    ylim([MI,MA]);
    xlabel('$x$');
    ylabel('$u$');
    title('Kuramoto-Sivashinsky Simulation');
    drawnow
    
    frame=getframe(h);
    im=frame2im(frame);
    [imind,cm]=rgb2ind(im,256);
    if k==1
        imwrite(imind,cm,filename,'gif','DelayTime',0,'Loopcount',inf);
    else
        imwrite(imind,cm,filename,'gif','DelayTime',0,'WriteMode','append');
    end
end

end