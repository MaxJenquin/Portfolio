function coeffs = p6s( a, b )
% polynomial coefficients for sine term in SM kernel 6th spatial deriv

% returns coeffs on: (tau^5, tau^3, tau) in that order

coeffs = zeros(3,1);
coeffs(1) = -192*a^5*b; %tau^5
coeffs(2) = -960*a^4*b+160*a^3*b^3; %tau^3
coeffs(3) = -960*a^3*b+240*a^2*b^3-12*a*b^5; %tau

end
