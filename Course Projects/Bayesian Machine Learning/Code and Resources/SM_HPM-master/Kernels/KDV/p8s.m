function coeffs = p8s( a, b )
% polynomial coefficients for sine term in SM kernel 8th spatial deriv

% returns coeffs on: (tau^7, tau^5, tau^3, tau) in that order

coeffs = zeros(4,1);
coeffs(1) = -1104*a^7*b; %tau^7
coeffs(2) = -10992*a^6*b+1792*a^5*b^3; %tau^5
coeffs(3) = -27840*a^5*b+8960*a^4*b^3-448*a^3*b^5; %tau^3
coeffs(4) = -14880*a^4*b+6960*a^3*b^3-672*a^2*b^5+16*a*b^7; %tau

end
