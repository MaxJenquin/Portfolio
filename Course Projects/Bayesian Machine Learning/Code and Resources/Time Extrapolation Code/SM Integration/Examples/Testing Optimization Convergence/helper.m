errors = zeros(6,100);
for nmix=1:6
    for k=1:100
        errors(nmix,k) = KDV_SM_fit_exploration(nmix);
    end    
end
save('errors.mat','errors');
