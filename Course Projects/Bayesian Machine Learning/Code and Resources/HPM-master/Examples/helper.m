clc;
close all;
[diffs, times, ndiffs, ntimes] = KS_Sample_Test();
save('overnight.mat');

%x = load('overnight.mat');
%diffs=x.diffs;
%etc.
