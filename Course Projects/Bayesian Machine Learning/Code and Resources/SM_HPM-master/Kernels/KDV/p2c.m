function coeffs = p2c( a, b )
% polynomial coefficients for cosine term in SM kernel 2nd spatial deriv

% returns coeffs on: (tau^2, tau^0) in that order

coeffs = zeros(2,1);
coeffs(1) = 4*a^2; %tau^2
coeffs(2) = 2*a-b^2; %tau^0

end