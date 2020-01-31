function coeffs = p3s( a, b )
% polynomial coefficients for sine term in SM kernel 3rd spatial deriv

% returns coeffs on: (tau^2, tau^0) in that order

coeffs = zeros(2,1);
coeffs(1) = -12*a^2*b; %tau^2
coeffs(2) = -6*a*b+b^3; %tau^0

end