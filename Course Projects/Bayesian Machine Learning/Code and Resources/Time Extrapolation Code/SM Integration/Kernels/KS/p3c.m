function coeffs = p3c( a, b )
% polynomial coefficients for cosine term in SM kernel 3rd spatial deriv

% returns coeffs on: (tau^3, tau) in that order

coeffs = zeros(2,1);
coeffs(1) = 8*a^3; %tau^3
coeffs(2) = 12*a^2-6*a*b^2; %tau

end