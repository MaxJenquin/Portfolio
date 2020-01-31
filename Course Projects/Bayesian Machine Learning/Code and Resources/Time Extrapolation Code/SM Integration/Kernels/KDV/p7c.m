function coeffs = p7c( a, b )
% polynomial coefficients for cosine term in SM kernel 7th spatial deriv

% returns coeffs on: (tau^7, tau^5, tau^3, tau) in that order

coeffs = zeros(4,1);
coeffs(1) = 128*a^7; %tau^7
coeffs(2) = 1344*a^6-672*a^5*b^2; %tau^5
coeffs(3) = 3360*a^5-3360*a^4*b^2+280*a^3*b^4; %tau^3
coeffs(4) = 1680*a^4-2760*a^3*b^2+420*a^2*b^4-14*a*b^6; %tau

end
