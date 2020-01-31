function coeffs = p6c( a, b )
% polynomial coefficients for cosine term in SM kernel 6th spatial deriv

% returns coeffs on: (tau^6, tau^4, tau^2, tau^0) in that order

coeffs = zeros(4,1);
coeffs(1) = 64*a^6; %tau^6
coeffs(2) = 480*a^5-240*a^4*b^2; %tau^4
coeffs(3) = 720*a^4-720*a^3*b^2+60*a^2*b^4; %tau^2
coeffs(4) = 120*a^3-180*a^2*b^2+30*a*b^4-b^6; %tau^0

end
