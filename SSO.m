function y = SSO(d, x, sigma, type, algo, eta)
% Stochastic Second Order Oracle
% type 1 = Multimodal function

temp = zeros(d,1);

if type == 1 % Multimodal function
    if strcmp(algo, 'ub')% Unbiased Hessian
        %   g_2(x) = (sin((pi*x)/20)^5*(ln(2)*(x-10)*sin((pi*x)/20)-480*pi*cos((pi*x)/20)))/(25*2^((x^2-20*x+19300)/3200))
        %   gg_2(x) =  (sin((pi*x)/20)^4*((ln(2)*(ln(2)*(x-10)^2-1600)-38400*pi^2)*sin((pi*x)/20)^2-960*pi*ln(2)*(x-10)*cos((pi*x)/20)*sin((pi*x)/20)+192000*pi^2*cos((pi*x)/20)^2))/(625*2^((x^2-20*x+38500)/3200))

        %Gaussian zero mean noise
        noise = sigma*randn(d,d);
        
        temp = zeros(d,d);
        for i = 1:d
            temp(i,i) = -(sin((pi*x(i))/20)^4*((log(2)*(log(2)*(x(i)-10)^2-1600)-38400*pi^2)*sin((pi*x(i))/20)^2-960*pi*log(2)*(x(i)-10)*cos((pi*x(i))/20)*sin((pi*x(i))/20)+192000*pi^2*cos((pi*x(i))/20)^2))/(625*2^((x(i)^2-20*x(i)+38500)/3200));
        end
        
        y = temp + noise;
        
    elseif strcmp(algo,'gs') % Biased Hessian using Gaussian smoothing
        
        % Generating perturbation
        delta = randn(d,1);
        
        x_plus = x + eta*delta;
        x_minus = x - eta*delta;
        
        y_plus = SZO(d, x_plus, sigma, type);
        y_minus = SZO(d, x_minus, sigma, type);
        y_0 = SZO(d, x, sigma, type);

        
        M_n = delta*delta' - eye(d);
        y = ((y_plus+y_minus-2*y_0)/(2*eta^2))*M_n;
        
        
    elseif strcmp(algo,'spsa') % Biased Hessian using 2SPSA
        
        % Generating perturbation
        delta = 2*round(rand(d,1))-1;
        y = zeros(d,d);
        
        x_plus = x + eta*delta;
        x_minus = x - eta*delta;
        
        y_plus = SZO(d, x_plus, sigma, type);
        y_minus = SZO(d, x_minus, sigma, type);

        delta_tilde = 2*round(rand(d,1))-1;
        
        x_plus_tilde = x_plus + eta*delta_tilde;
        x_minus_tilde = x_minus - eta*delta_tilde;
        
        y_plus_tilde = SZO(d, x_plus_tilde, sigma, type);
        y_minus_tilde = SZO(d, x_minus_tilde, sigma, type);

        ghatplus=(y_plus_tilde-y_plus)./(eta*delta_tilde);
        ghatminus=(y_minus_tilde-y_minus)./(eta*delta_tilde);
        deltaghat=ghatplus-ghatminus;
        for i=1:d
            y(:,i)=deltaghat(i)./(2*eta*delta);
        end
        
    elseif strcmp(algo,'rdsa_u') % Biased Hessian using 2RDSA_Uniform
        u = 1;
        % Generate uniform [-u,u] perturbations
        delta = unifrnd(-u,u,d,1);
        
        x_plus = x + eta*delta;
        x_minus = x - eta*delta;
        
        y_plus = SZO(d, x_plus, sigma, type);
        y_minus = SZO(d, x_minus, sigma, type);
        y_0 = SZO(d, x, sigma, type);
    
        % GENERATE THE HESSIAN UPDATE
        M_n = delta*delta';
        for idx=1:d
            M_n(idx,idx) = 5/2*(M_n(idx,idx) - 1/3);
        end      

        % STATEMENT PROVIDING AN AVERAGE OF SP GRAD. APPROXS. PER ITERATION      
        y = 9/2.*((y_plus+y_minus-2*y_0)/(eta^2)).*M_n;
        
     elseif strcmp(algo,'rdsa_ab') % Biased Hessian using 2RDSA_AsymBer
        
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
        y_0 = SZO(d, x, sigma, type);


        % GENERATE THE HESSIAN UPDATE
        M_n = delta*delta'*(1/(2*(1+epsilon)^2));
        beta = (1+epsilon)*(1+ (1+epsilon)^3)/(2+epsilon);
        kappa = beta * (1 - (1+epsilon)^2/beta);
        for idx=1:d
            M_n(idx,idx) = 1/kappa*((2*(1+epsilon)^2)*M_n(idx,idx) - (1+epsilon));
        end      
        
        % STATEMENT PROVIDING AN AVERAGE OF SP GRAD. APPROXS. PER ITERATION      
        y = ((y_plus+y_minus-2*y_0)/(eta^2))*M_n;
        
        

      elseif strcmp(algo,'rdsa_lex') % Biased Hessian using 2RDSA_Lex_DP
          
        % Generating lexicograpic sequence  
        delta = zeros(3^d,d);
        for t = 1:d 
           temp = [-1*ones(2*3^(d-t),1); 2*ones(3^(d-t),1)]; 
           delta(:,t) = repmat(temp,3^(t-1),1);
        end
        y = zeros(d,d);

        for j = 1:3^d

            % GENERATE THE HESSIAN UPDATE
            M_n = delta(j,:)'*delta(j,:);
            beta =  2*3^d;
            kappa = (1/(2*3^(d-1)))-1;
            for idx=1:d
            M_n(idx,idx) = 1/kappa*(M_n(idx,idx) - beta);
            end   
            
            x_plus = x + eta*delta(j,:)';
            x_minus = x - eta*delta(j,:)';
            
            y_plus = SZO(d, x_plus, sigma, type);
            y_minus = SZO(d, x_minus, sigma, type); 
            y_0 = SZO(d, x, sigma, type);

            
            y = y + M_n*((y_plus+y_minus-2*y_0)/(eta^2));
        end

        y = y/(beta)^2;
        
       elseif strcmp(algo,'rdsa_perm') % Biased Hessian using 2RDSA_Perm_DP 
           
         % Generating permutation matrix
        delta = eye(d);
        delta = delta(randperm(d),:);
        y = zeros(d,d);

        for j = 1:d
            % GENERATE THE HESSIAN UPDATE
            M_n = delta(j,:)'*delta(j,:);

            x_plus = x + eta*delta(j,:)';
            x_minus = x - eta*delta(j,:)';
            
            y_plus = SZO(d, x_plus, sigma, type);
            y_minus = SZO(d, x_minus, sigma, type); 
            y_0 = SZO(d, x, sigma, type);

            
            y = y + M_n*((y_plus+y_minus-2*y_0)/(eta^2));
        end
       
      
        
    end
        
end

end
