% This function implements Zeroth-order Randomized Stochastic Gradient (ZRSG) algorithm with unbiased and biased
% gradient estimates for non-convex support vector machine problem.

% Parameters:
% algo -> ub - unbiased, spsa, rdsa_u, rdsa_ab, rdsa_perm, rdsa_lex,
% rdsa_kw
% N -> Iteration limit
% T -> Number of independent replications
% Q -> Number of samples to find quality of solution at x_R
% x_1 -> Initial point
% gamma -> Step size (vector for all k = 1,...,N)
% P_R -> Probability mass function (vector for all k = 1,...,N)
% eta -> Perturbation constant
% lambda -> Constant Lagrange multiplier
% u -> Training features
% v -> Training true lables 
% test_u -> Testing features
% test_v -> Testing true lables 

function [all] = rsg_svm(algo, N, T, Q, x_1, gamma, P_R, eta, lambda, u, v, test_u, test_v)

d = size(x_1,1); % Dimension of the problem

% Setting bound on x
x_lo=-10.*ones(d,1);   %lower bounds on x  
x_hi=10.*ones(d,1);  %upper bounds on x 
    
sng = zeros(T,1);
test_acc = zeros(T,1);

    
for i = 1:T % loop for T replications
    x_k = x_1;

    R = randsample(N, 1, true, P_R); % Get randon variable R from probability mass function P_R

    for k = 1:R % Loop for random R iteration
      grad = SVM( x_k, u(randi([1 size(u,1)]),:)', v(randi([1 size(u,1)])), lambda, eta(k), algo, 1 ); % Estimate gradient
      x_k = x_k - gamma(k)*grad; % Stochastic Gradient descent
      % Project theta onto a bounded set, component-wise
      x_k=min(x_k,x_hi);
      x_k=max(x_k,x_lo); 
    end

    x_R = x_k; % Output x_R
    
    % Estimating gradient at x_R from Q i.i.d. samples
    grad_x_R = 0; %zeros(d,1);
    iid_samples_index = randi([1 size(test_u,1)],Q,1); % Getting random Q samples index
    u_iid = test_u(iid_samples_index,:);
    v_iid = test_v(iid_samples_index,:);

    for j = 1:Q % Loop for Q i.i.d. samples to estimate gradient of x_R
        grad_x_R = grad_x_R + SVM( x_R, u_iid(j,:)', v_iid(j), lambda, eta(k), algo, 1 ); % Get gradient of x_R
    end
    grad_x_R = grad_x_R./Q;
    sng(i) = norm(grad_x_R,2)^2; % Squared norm of gradient at x_R
    
    % Calculating classification accuracy on testing data
    estimated_v = ones(size(test_v,1),1);
    for l = 1:size(test_v,1)
        if x_R'*test_u(l,:)' <= 0 
            estimated_v(l) = -1;
        end
    end

    test_acc(i) = sum(estimated_v == test_v)*100/size(test_v,1);

end


% Display results: sng, misclassification (testing) error
str = sprintf('SNG: %f +- %f', mean(sng), std(sng)/sqrt(T)); disp(str);
str = sprintf('Classification accuracy: %f +- %f \n', mean(test_acc), std(test_acc)/sqrt(T)); disp(str);


all = [mean(sng), std(sng)/sqrt(T), mean(test_acc), std(test_acc)/sqrt(T)];
end


