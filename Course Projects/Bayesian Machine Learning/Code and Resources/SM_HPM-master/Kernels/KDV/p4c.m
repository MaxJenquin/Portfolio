function coeffs = p4c( a, b )
% polynomial coefficients for cosine term in SM kernel 4th spatial deriv

% returns coeffs on: (tau^4, tau^2, tau^0) in that order

coeffs = zeros(3,1);
coeffs(1) = 16*a^4; %tau^4
coeffs(2) = 48*a^3-24*a^2*b^2; %tau^2
coeffs(3) = 12*a^2-12*a*b^2+b^4; %tau^0

end