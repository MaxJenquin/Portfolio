function coeffs = p4s( a, b )
% polynomial coefficients for sine term in SM kernel 4th spatial deriv

% returns coeffs on: (tau^3, tau) in that order

coeffs = zeros(2,1);
coeffs(1) = -32*a^3*b; %tau^3
coeffs(2) = -48*a^2*b+8*a*b^3; %tau

end