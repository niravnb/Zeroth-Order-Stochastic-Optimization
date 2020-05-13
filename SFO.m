function grad = SFO(d, x, sigma, type, algo, eta)
% Stochastic First Order Oracles
% type 1 = Multimodal function



temp = zeros(d,1);

if type == 1 % Multimodal function
    if strcmp(algo,'ub') % Unbiased gradient
        %   g_2(x) = (sin((pi*x)/20)^5*(ln(2)*(x-10)*sin((pi*x)/20)-480*pi*cos((pi*x)/20)))/(25*2^((x^2-20*x+19300)/3200))
        
        %Gaussian zero mean noise
        noise = sigma*randn(d,1);
        
        for i = 1:d
           temp(i) = (sin((pi*x(i))/20)^5*(log(2)*(x(i)-10)*sin((pi*x(i))/20)-480*pi*cos((pi*x(i))/20)))/(25*2^((x(i)^2-20*x(i)+19300)/3200));
        end
        
        grad = temp + noise;
        
    elseif strcmp(algo,'gs') % Biased gradient using Gaussian smoothing
        
        % Generating perturbation
        delta = randn(d,1);
        
        x_plus = x + eta.*delta;
        x_minus = x;
        
        y_plus = SZO(d, x_plus, sigma, type);
        y_minus = SZO(d, x_minus, sigma, type);
        
        grad = ((y_plus - y_minus)/eta)*delta;
        
    elseif strcmp(algo,'spsa') % Biased gradient using 1SPSA
        
        % Generating perturbation
        delta = 2*round(rand(d,1))-1;
        
        x_plus = x + eta*delta;
        x_minus = x - eta*delta;
        
        y_plus = SZO(d, x_plus, sigma, type);
        y_minus = SZO(d, x_minus, sigma, type);
        
        grad = (y_plus - y_minus)./(2*eta.*delta);
        
    elseif strcmp(algo,'rdsa_u') % Biased gradient using 1RDSA_Uniform
        u = 1;
        % Generate uniform [-u,u] perturbations
        delta = unifrnd(-u,u,d,1);
        
        x_plus = x + eta*delta;
        x_minus = x - eta*delta;
        
        y_plus = SZO(d, x_plus, sigma, type);
        y_minus = SZO(d, x_minus, sigma, type);
        
        grad = 3*((y_plus - y_minus)/(2*eta))*delta;
        
     elseif strcmp(algo,'rdsa_ab') % Biased gradient using 1RDSA_AsymBer
        
        % Generating Asymmetric Bernoulli perturbation
        epsilon = 0.0001;
        delta = zeros(d,1);
        unifrands = unifrnd(0,1,d,1);
        for j=1:d
            if unifrands(j,1) < ((1+epsilon)/(2+epsilon))
                delta(j,1) = -1;
            else
                delta(j,1) = 1+epsilon;
            end
        end
        
        x_plus = x + eta*delta;
        x_minus = x - eta*delta;
        
        y_plus = SZO(d, x_plus, sigma, type);
        y_minus = SZO(d, x_minus, sigma, type);
        
        grad = (1/(1+epsilon))*((y_plus - y_minus)/(2*eta))*delta;
        
      elseif strcmp(algo,'rdsa_lex') % Biased gradient using 1RDSA_Lex_DP
          
        % Generating lexicograpic sequence  
        delta = zeros(3^d,d);
        for t = 1:d 
           temp = [-1*ones(2*3^(d-t),1); 2*ones(3^(d-t),1)]; 
           delta(:,t) = repmat(temp,3^(t-1),1);
        end
        grad = zeros(d,1);

        for j = 1:3^d
            
            x_plus = x + eta*delta(j,:)';
            x_minus = x - eta*delta(j,:)';
            
            y_plus = SZO(d, x_plus, sigma, type);
            y_minus = SZO(d, x_minus, sigma, type); 
            
            grad = grad + delta(j,:)'*((y_plus - y_minus)/(2*eta));
        end

        grad = grad/(2*3^d);
        
       elseif strcmp(algo,'rdsa_perm') % Biased gradient using 1RDSA_Perm_DP 
           
         % Generating permutation matrix
        delta = eye(d);
        delta = delta(randperm(d),:);
        grad = zeros(d,1);

        for j = 1:d
            
            x_plus = x + eta*delta(j,:)';
            x_minus = x - eta*delta(j,:)';
            
            y_plus = SZO(d, x_plus, sigma, type);
            y_minus = SZO(d, x_minus, sigma, type); 
            
            grad = grad + delta(j,:)'*((y_plus - y_minus)/(2*eta));
        end
       
      elseif strcmp(algo,'rdsa_kw') % Biased gradient using 1RDSA_kw_DP 
          
        % Generating permutation matrix
        delta = eye(d);
        grad = zeros(d,1);

        for j = 1:d
            x_plus = x + eta*delta(j,:)';
            x_minus = x - eta*delta(j,:)';
            
            y_plus = SZO(d, x_plus, sigma, type);
            y_minus = SZO(d, x_minus, sigma, type); 
            
            grad(j,1) = (y_plus - y_minus)/(2*eta);
        end
        
    end
        
end

end
