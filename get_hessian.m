function H = get_hessian(d, x, type)
% returns hessian for type 1 = Multimodal function

if type == 1 % Multimodal function
%   g_2(x) = (sin((pi*x)/20)^5*(ln(2)*(x-10)*sin((pi*x)/20)-480*pi*cos((pi*x)/20)))/(25*2^((x^2-20*x+19300)/3200))
%   gg_2(x) =  (sin((pi*x)/20)^4*((ln(2)*(ln(2)*(x-10)^2-1600)-38400*pi^2)*sin((pi*x)/20)^2-960*pi*ln(2)*(x-10)*cos((pi*x)/20)*sin((pi*x)/20)+192000*pi^2*cos((pi*x)/20)^2))/(625*2^((x^2-20*x+38500)/3200))
    temp = zeros(d,d);
    for i = 1:d
      temp(i,i) = -(sin((pi*x(i))/20)^4*((log(2)*(log(2)*(x(i)-10)^2-1600)-38400*pi^2)*sin((pi*x(i))/20)^2-960*pi*log(2)*(x(i)-10)*cos((pi*x(i))/20)*sin((pi*x(i))/20)+192000*pi^2*cos((pi*x(i))/20)^2))/(625*2^((x(i)^2-20*x(i)+38500)/3200));
    end
    H = temp;
 end
end
