% This function implements Zeroth-order Randomized Stochastic quasi-Newton (ZRSQN) algorithm with unbiased and biased
% gradient estimates for Multimodal function.

% Parameters:
% d -> Dimension of the problem
% type -> 1 = Multimodal function
% algo -> ub - unbiased, spsa, rdsa_u, rdsa_ab, rdsa_perm, rdsa_lex,
% rdsa_kw
% N -> Iteration limit
% T -> Number of independent replications
% Q -> Number of samples to find quality of solution at x_R
% x_1 -> Initial point
% x_star -> Optimal point
% gamma -> Step size (vector for all k = 1,...,N)
% P_R -> Probability mass function (vector for all k = 1,...,N)
% eta -> Perturbation constant
% sigma -> Noise in SFO or SZO

function [all] = rsqn(d, type, algo, N, T, Q, x_1, x_star, gamma, P_R, eta, sigma)

nmse_x_1 = norm((x_1-x_star)); % Initial MSE

% Setting bound for x
if type == 1
    x_lo=0*ones(d,1);   %lower bounds on x  
    x_hi=100*ones(d,1);  %upper bounds on x 
end

sng = zeros(T,1);
f_diff = zeros(T,1);
nmse = zeros(T,1);

    
for i = 1:T % Loop for T replications
    x_k = x_1;
    R = randsample(N, 1, true, P_R); % Get randon variable R from probability mass function P_R
%     Hbar = 500*eye(d);
    for k = 1:R % Loop for random R iteration
      grad = SFO(d, x_k, sigma, type, algo, eta(k)); % Estimate gradient
      Hhat = SSO( d, x_k, sigma, type, algo, eta(k)); % Estimate Hessian
      
      % STATEMENT PROVIDING AN AVERAGE OF SP GRAD. APPROXS. PER ITERATION      
      Hhat=.5*(Hhat+Hhat');  
      if k == 1
          Hbar = Hhat;
      else
          Hbar=(k/(k+1))*Hbar+Hhat/(k+1);  
      end

%        Hbar=(k/(k+1))*Hbar+Hhat/(k+1);  

        
      %   THE THETA UPDATE (FORM BELOW USES GAUSSIAN ELIMINATION TO AVOID DIRECT 
      %   COMPUTATION OF HESSIAN INVERSE)
      H_k=sqrtm(Hbar*Hbar+.000001*eye(d)/(k+1));
    
      x_k = x_k - gamma(k)*(H_k\grad); % Stochastic quasi-Newton descent

      if type == 1
         % Project theta onto a bounded set, component-wise
        x_k=min(x_k,x_hi);
        x_k=max(x_k,x_lo); 
      end
    end

    x_R = x_k; % Output x_R
    
    grad_x_R = zeros(d,1);
    for q = 1:Q % Loop for Q times to estimate gradient at x_R
        grad_x_R = grad_x_R + SFO(d, x_R, sigma, type, algo, eta(k)); % Get gradient of x_R
    end
    grad_x_R = grad_x_R./Q;
    sng(i) = norm(grad_x_R)^2; % Squared norm of gradient at x_R

    f_x_R = SZO(d, x_R, 0, type);
    f_x_star = SZO(d, x_star, 0, type);
    f_diff(i) = abs(f_x_R - f_x_star); % |f(x_R) - f(x^*)|

    nmse(i) = norm((x_R-x_star))/nmse_x_1; % NMSE |x_R - x^*| / |x_1 - x^*|

end

% Display results: sng, f-diff, normalized mean square error
str = sprintf('SNG: %f +- %f', mean(sng), std(sng)/sqrt(T)); disp(str);
% str = sprintf('f_diff: %f +- %f', mean(f_diff), std(f_diff)/sqrt(T)); disp(str);
% str = sprintf('Normalised MSE: %f +- %f', mean(nmse), std(nmse)/sqrt(T)); disp(str);

all = [mean(sng), std(sng)/sqrt(T), mean(f_diff), std(f_diff)/sqrt(T), mean(nmse), std(nmse)/sqrt(T)];

end


