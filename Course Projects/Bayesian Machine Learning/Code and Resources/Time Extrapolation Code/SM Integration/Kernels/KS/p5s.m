function coeffs = p5s( a, b )
% polynomial coefficients for sine term in SM kernel 5th spatial deriv

% returns coeffs on: (tau^4, tau^2, tau^0) in that order

coeffs = zeros(3,1);
coeffs(1) = -80*a^4*b; %tau^4
coeffs(2) = -240*a^3*b+40*a^2*b^3; %tau^2
coeffs(3) = -60*a^2*b+20*a*b^3-b^5; %tau^0

end