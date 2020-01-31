function coeffs = p7s( a, b )
% polynomial coefficients for sine term in SM kernel 7th spatial deriv

% returns coeffs on: (tau^6, tau^4, tau^2, tau^0) in that order

coeffs = zeros(4,1);
coeffs(1) = -488*a^6*b; %tau^6
coeffs(2) = -3360*a^5*b+560*a^4*b^3; %tau^4
coeffs(3) = -5520*a^4*b+1680*a^3*b^3-84*a^2*b^5; %tau^2
coeffs(4) = -1080*a^3*b+420*a^2*b^3-42*a*b^5+b^7; %tau^0

end
