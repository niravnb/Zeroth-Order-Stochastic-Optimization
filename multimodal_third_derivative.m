function [ y ] = multimodal_third_derivative( x )
%Returns third derivative of multimodal function

y = -(2.^(-x.^2/3200+x/160-577/32).*sin((pi.*x)/20).^3.*((log(2).^3.*x.^3-30.*log(2).^3.*x.^2+(300.*log(2).^3-4800.*log(2).^2-691200.*pi.^2.*log(2)).*x-1000.*log(2).^3+48000.*log(2).^2+6912000.*pi.^2.*log(2)).*sin((pi.*x)/20).^3+(-1440.*pi.*log(2).^2.*x.^2+28800.*pi.*log(2).^2.*x-144000.*pi.*log(2).^2+2304000.*pi.*log(2)+110592000.*pi.^3).*cos((pi.*x)/20).*sin((pi.*x)/20).^2+(576000.*pi.^2.*log(2).*x-5760000.*pi.^2.*log(2)).*sin((pi.*x)/20)-61440000.*pi.^3.*cos((pi.*x)/20)))/15625;


end

