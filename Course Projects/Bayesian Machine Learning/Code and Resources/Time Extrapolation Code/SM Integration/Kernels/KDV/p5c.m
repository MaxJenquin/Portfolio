function coeffs = p5c( a, b )
% polynomial coefficients for cosine term in SM kernel 5th spatial deriv

% returns coeffs on: (tau^5, tau^3, tau) in that order

coeffs = zeros(3,1);
coeffs(1) = 32*a^5; %tau^5
coeffs(2) = 160*a^4-80*a^3*b^2; %tau^3
coeffs(3) = 120*a^3-120*a^2*b^2+10*a*b^4; %tau

end
