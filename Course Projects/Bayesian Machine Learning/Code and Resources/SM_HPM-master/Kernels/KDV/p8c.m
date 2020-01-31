function coeffs = p8c( a, b )
% polynomial coefficients for cosine term in SM kernel 8th spatial deriv

% returns coeffs on: (tau^8, tau^6, tau^4, tau^2, tau^0) in that order

coeffs = zeros(5,1);
coeffs(1) = 256*a^8; %tau^8
coeffs(2) = 3584*a^7-1832*a^6*b^2; %tau^6
coeffs(3) = 13440*(a^6-a^5*b^2)+ 1120*a^4*b^4; %tau^4
coeffs(4) = 13440*a^5-21120*a^4*b^2+3360*a^3*b^4-112*a^2*b^6; %tau^2
coeffs(5) = 1680*a^4-3840*a^3*b^2+840*a^2*b^4-56*a*b^6+b^8; %tau^0

end
