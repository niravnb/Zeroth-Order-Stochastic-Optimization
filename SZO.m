function y = SZO(d, x, sigma, type)
% Stochastic Zeroth Order Oracle
% type 1 = Multimodal function

%noise with one extra dimension
z = sigma*randn(d+1,1);
noise = [x' 1]*z;

if type == 1 % Multimodal function
%    f_2(x) = sin(0.05 pi x)^6/2^(2((x-10)/80)^2)
%    f(x) = -[f_2(x_1) + f_2(x_2)] + 2 + noise
    temp = 0;
    for i = 1:d
       temp = temp + sin(0.05*pi*x(i))^6/2^(2*((x(i)-10)/80)^2);
    end
    y = -temp + d + noise;
elseif type == 2 % SVM
    % f(x) = 1 - tanh(v*x'*u) + lambda*norm(x)^2
    
    y = 1 - tanh(v*x'*u) + lambda*norm(x)^2;
end
