function [ y ] = SVM( x, u, v, lambda, eta, algo, order )
%Returns nonconvex support vector machine (SVM) sigmoid loss function value,
% gradient or Hessian depending on argument passed.

d = size(x,1);

if order == 0 % Function value

    y = 1 - tanh(v*x'*u) + lambda*norm(x)^2;

elseif order == 1 % Gradient
    
    if strcmp(algo,'ub') % Unbiased gradient
        y = -v*sech(v*x'*u)^2*u + 2*lambda*x;
        
    elseif strcmp(algo,'gs') % Biased gradient using Gaussian smoothing
        
        % Generating perturbation
        delta = randn(d,1);
        
        x_plus = x + eta.*delta;
        x_minus = x;
        
        y_plus = SVM(x_plus, u, v, lambda, eta, algo, 0);
        y_minus = SVM(x_minus, u, v, lambda, eta, algo, 0);
        
        y = ((y_plus - y_minus)/eta)*delta;
        
    elseif strcmp(algo,'spsa') % Biased gradient using 1SPSA
        
        % Generating perturbation
        delta = 2*round(rand(d,1))-1;
        
        x_plus = x + eta*delta;
        x_minus = x - eta*delta;
        
        y_plus = SVM(x_plus, u, v, lambda, eta, algo, 0);
        y_minus = SVM(x_minus, u, v, lambda, eta, algo, 0);
        
        y = (y_plus - y_minus)./(2*eta.*delta);
        
    elseif strcmp(algo,'rdsa_u') % Biased gradient using 1RDSA_Uniform
        unif_const = 1;
        % Generate uniform [-u,u] perturbations
        delta = unifrnd(-unif_const,unif_const,d,1);
        
        x_plus = x + eta*delta;
        x_minus = x - eta*delta;
        
        y_plus = SVM(x_plus, u, v, lambda, eta, algo, 0);
        y_minus = SVM(x_minus, u, v, lambda, eta, algo, 0);
        
        y = 3*((y_plus - y_minus)/(2*eta))*delta;
        
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
        
        y_plus = SVM(x_plus, u, v, lambda, eta, algo, 0);
        y_minus = SVM(x_minus, u, v, lambda, eta, algo, 0);
        
        y = (1/(1+epsilon))*((y_plus - y_minus)/(2*eta))*delta;
        
      elseif strcmp(algo,'rdsa_lex') % Biased gradient using 1RDSA_Lex_DP
          
        % Generating lexicograpic sequence  
        delta = zeros(3^d,d);
        for t = 1:d 
           temp = [-1*ones(2*3^(d-t),1); 2*ones(3^(d-t),1)]; 
           delta(:,t) = repmat(temp,3^(t-1),1);
        end
        y = zeros(d,1);

        for j = 1:3^d
            
            x_plus = x + eta*delta(j,:)';
            x_minus = x - eta*delta(j,:)';
            
            y_plus = SVM(x_plus, u, v, lambda, eta, algo, 0);
            y_minus = SVM(x_minus, u, v, lambda, eta, algo, 0);
            
            y = y + delta(j,:)'*((y_plus - y_minus)/(2*eta));
        end

        y = y/(2*3^d);
        
       elseif strcmp(algo,'rdsa_perm') % Biased gradient using 1RDSA_Perm_DP 
           
         % Generating permutation matrix
        delta = eye(d);
        delta = delta(randperm(d),:);
        y = zeros(d,1);

        for j = 1:d
            
            x_plus = x + eta*delta(j,:)';
            x_minus = x - eta*delta(j,:)';
            
            y_plus = SVM(x_plus, u, v, lambda, eta, algo, 0);
            y_minus = SVM(x_minus, u, v, lambda, eta, algo, 0);
            
            y = y + delta(j,:)'*((y_plus - y_minus)/(2*eta));
        end
       
      elseif strcmp(algo,'rdsa_kw') % Biased gradient using 1RDSA_kw_DP 
          
        % Generating permutation matrix
        delta = eye(d);
        y = zeros(d,1);

        for j = 1:d
            x_plus = x + eta*delta(j,:)';
            x_minus = x - eta*delta(j,:)';
            
            y_plus = SVM(x_plus, u, v, lambda, eta, algo, 0);
            y_minus = SVM(x_minus, u, v, lambda, eta, algo, 0);
            
            y(j,1) = (y_plus - y_minus)/(2*eta);
        end
        
    end    
    
elseif order == 2 % Hessian
    
   if strcmp(algo,'ub') % Unbiased Hessian
        y = 2*v^2*sech(v*x'*u)^2*tanh(v*x'*u)*(u*u') + 2*lambda*eye(size(u,2));
        
    elseif strcmp(algo,'gs') % Biased Hessian using Gaussian smoothing
        
        % Generating perturbation
        delta = randn(d,1);
        
        x_plus = x + eta.*delta;
        x_minus = x - eta*delta;
        
        y_plus = SVM(x_plus, u, v, lambda, eta, algo, 0);
        y_minus = SVM(x_minus, u, v, lambda, eta, algo, 0);
        y_0 = SVM(x, u, v, lambda, eta, algo, 0);

        
        M_n = delta*delta' - eye(d);
        y = ((y_plus+y_minus-2*y_0)/(2*eta^2))*M_n;
        
        

    elseif strcmp(algo,'spsa') % Biased Hessian using 2SPSA
        
        % Generating perturbation
        delta = 2*round(rand(d,1))-1;
        y = zeros(d,d);
        
        x_plus = x + eta*delta;
        x_minus = x - eta*delta;
        
        y_plus = SVM(x_plus, u, v, lambda, eta, algo, 0);
        y_minus = SVM(x_minus, u, v, lambda, eta, algo, 0);

        delta_tilde = 2*round(rand(d,1))-1;
        
        x_plus_tilde = x_plus + eta*delta_tilde;
        x_minus_tilde = x_minus - eta*delta_tilde;
        
        y_plus_tilde = SVM(x_plus_tilde, u, v, lambda, eta, algo, 0);
        y_minus_tilde = SVM(x_minus_tilde, u, v, lambda, eta, algo, 0);

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
%         y = zeros(d,d);
        
        x_plus = x + eta*delta;
        x_minus = x - eta*delta;
        
        y_plus = SVM(x_plus, u, v, lambda, eta, algo, 0);
        y_minus = SVM(x_minus, u, v, lambda, eta, algo, 0);
        y_0 = SVM(x, u, v, lambda, eta, algo, 0);
    
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
        
        y_plus = SVM(x_plus, u, v, lambda, eta, algo, 0);
        y_minus = SVM(x_minus, u, v, lambda, eta, algo, 0);
        y_0 = SVM(x, u, v, lambda, eta, algo, 0);


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
            
            y_plus = SVM(x_plus, u, v, lambda, eta, algo, 0);
            y_minus = SVM(x_minus, u, v, lambda, eta, algo, 0);
            y_0 = SVM(x, u, v, lambda, eta, algo, 0);

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
            
            y_plus = SVM(x_plus, u, v, lambda, eta, algo, 0);
            y_minus = SVM(x_minus, u, v, lambda, eta, algo, 0);
            y_0 = SVM(x, u, v, lambda, eta, algo, 0);

            y = y + M_n*((y_plus+y_minus-2*y_0)/(eta^2));
        end
   else
       y = 2*v^2*sech(v*x'*u)^2*tanh(v*x'*u)*(u*u') + 2*lambda*eye(size(u,2));
   end
elseif order == 3 % Third derivative
%     u_nonzero = u(u > 0); % selecting non zero values of u
%     perms = permn(u_nonzero,3); % all 3 permutations of non zero u
%     m = max(perms(:,1).*perms(:,2).*perms(:,3)); % selecting max of all 3 perm combinations of u
      m = max(abs(u))^3; % selecting max of all 3 perm combinations of u

    bracket_term = v*x'*u;
    y = abs(-4*v^3*m*sech(bracket_term)^2*tanh(bracket_term)^2 + 2*v^3*m*sech(bracket_term)^4);
        
end

end

