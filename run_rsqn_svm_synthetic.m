close all; clear all; clc
%% Input Parameters
    
ra = rng(105,'twister'); % Setting some random seed for reproducibility

d = 50; % Dimension of the problem
step = 0; % Setting type of stepsize; value of 0 mean min{(2C_l-1)/Lambda*C_u^2, 1/(d^2N)^2/3}, 1 means min{1/L, c/sqrt(N)}, 2 means 1\L, 3 means 1\LK with eta = 1

x_1 = 5.*unifrnd(0,1,d,1); % Initial point
lambda = 0.01; % Constant Lagrange multiplier

NList = [10, 250, 500, 1000, 1850, 3000, 5000]; % Iterations list
T = 50; % Replications

%% Generating Synthetic Dataset

dataset_len = 10000; % Setting dataset length
reduce_dim = 0; % Reduce dimension by SVD?

u = sprand(dataset_len,d,0.05);
x_bar = unifrnd(-1,1,d,1);

v = ones(dataset_len,1);
for i = 1:dataset_len
    if x_bar'*u(i,:)' < 0 
        v(i) = -1;
    end
end

%% Performing SVD to reduce dimension

if reduce_dim == 1
    original_norm = norm(u,'fro');
    [U,S,V,flag] = svds(u,min(size(u)));

    % Extract singular values
    singvals = diag(S);
    originalss = sum(singvals.^2);

    % Find out where to truncate the U, S, V matrices to get 90% accuracy
    tmp = 0;
    for i = 1:d
        tmp = tmp + singvals(i).^2;
        if tmp > 0.9*originalss
            newd = i;
            break
        end
    end

    reconstructed = U(:,1:newd)*S(1:newd,1:newd);
    reconstructed_norm = norm(reconstructed,'fro');
    err = abs(reconstructed_norm - original_norm)*100/original_norm;

    str = sprintf('Reduced dimension is %d and frobenious norm error is %f',newd,err); 
    disp(str);

    u = reconstructed;
    d = newd;
end

%% Splitting training and testing dataset  
training_len = 0.6*dataset_len;
test_len = 0.4*dataset_len;

training_u = u(1:training_len,:);
training_v = v(1:training_len);

test_u = u(training_len+1:end,:);
test_v = v(training_len+1:end);

%% Paramater Estimation L and sigma
N_0 = 200; % Using N_0 i.i.d. samples 
approx_samples = 200;

gradient_var = zeros(N_0,1);
B_list = zeros(N_0,1);
hessian_norm = zeros(N_0,1);
third_der_mag = zeros(N_0,1);
f_sigma = zeros(N_0,1);

x = randi(10).*unifrnd(-1,1,d,N_0); % Generating random points
initial = 0;
f_x_1 = 0;
iid_samples_index = randi([1 training_len],N_0,1);
u_iid = training_u(iid_samples_index,:);
v_iid = training_v(iid_samples_index,:);

for i = 1:N_0 % looping over N_0 random points x
    
    hessian = zeros(d,d);
    gradient_norm = zeros(approx_samples,1);
    grad_l2norm = 0;
    third_der = zeros(approx_samples,1);
    f_x = zeros(approx_samples,1);
   
    for j = 1:approx_samples % looping over N_0 iid samples (u,v)
       gradient = SVM( x(:,i), u_iid(j,:)', v_iid(j), lambda, 0, 'ub', 1 ); % Calling SFO
       gradient_norm(j) = norm(gradient,2)^2;
       grad_l2norm = grad_l2norm + norm(gradient,2);

       
       hessian = hessian + SVM( x(:,i), u_iid(j,:)', v_iid(j), lambda, 0, 0, 2 ); % Getting hessian
       
       third_der(j) = SVM( x(:,i), u_iid(j,:)', v_iid(j), lambda, 0, 0, 3 ); % Getting third derivative at x
       
       f_x(j) = SVM( x(:,i), u_iid(i,:)', v_iid(i), lambda, 0, 0, 0); % Calling SZO
    end
    
    gradient_var(i) = var(gradient_norm);
    B_list(i) = grad_l2norm/N_0;

    hessian = hessian./approx_samples;
    hessian_norm(i) = norm(hessian,2);
    
    third_der_mag(i) = max(third_der);
    
    f_sigma(i) = var(f_x);
    
    f_x_1 = f_x_1 + SVM( x_1, u_iid(i,:)', v_iid(i), lambda, 0, 0, 0);
   
end

% Estimate of f(x1)
f_x_1 = f_x_1/N_0;

Lambda = max(hessian_norm); % Estimate of Lambda
small_lambda = min(hessian_norm); % Estimate of lambda

C_l = 1/Lambda;
C_u = 1/small_lambda;

L = Lambda * C_u^2;
c_11 = C_u * d^3/Lambda;

sigma = max(gradient_var); % Estimate of sigma
f_sigma2 = max(f_sigma); % Estimate of noise in f
B = max(B_list); % Estimation of B

alpha_0 = max(third_der_mag); % Estimate of third derivative upper bound
alpha_1 = (2 + lambda*d*2^2)^2; % Estimate of E[f(x + eta*delta)^2] upper bound

%% Declaring variables to store results

approx_samples = test_len; % Samples to estimate f or gradient, % To find quality of solution at x_R


ub = [NList', zeros(length(NList),4)]; % Itr, mean_sng, std. err sng, classification acc. mean, classification acc. std. err
gs = [NList', zeros(length(NList),4)];
spsa = [NList', zeros(length(NList),4)];

rdsa_u = [NList', zeros(length(NList),4)];
rdsa_ab = [NList', zeros(length(NList),4)];


rdsa_lex = [NList', zeros(length(NList),4)];
rdsa_perm = [NList', zeros(length(NList),4)];

%% Saving dataset and parameter
% save(['synthetic_dataset']);

%% Loading dataset and parameter
% load(['synthetic_dataset']);

%% RSG_Unbiased algorithm
rng(ra);
algo = 'ub';
i = 1;
 disp('--------------------------------Starting Unbiased----------------------------------');
for N = NList
    str=sprintf('N = %d \n', N); disp(str);

    % Setting step size and Probability mass function
    eta = 0*ones(N,1);
    c = sqrt(2*(f_x_1)/(L*sigma));
        
    if step == 0
       gamma = ones(N,1)*min((2*C_l - 1)/L, c/(N^(2/3)));
    elseif step == 1
       gamma = ones(N,1)*min((2*C_l - 1)/L, c/(N^(2/3)));
    elseif step == 2
       gamma = ones(N,1).*(2*C_l - 1)/L;
    elseif step == 3
       gamma = 1/L.*1./(1:N); % Decreasing step size 
    end

    P_R = gamma./sum(gamma);
    
    ub(i,2:5) = rsqn_svm(algo, N, T, approx_samples, x_1, gamma, P_R, eta, lambda, training_u, training_v, test_u, test_v);
    i = i+1;
end
disp(ub(:,[2,4]));
disp('--------------------------------End Unbiased----------------------------------');

plot(NList,ub(:,2));
save(['RSQN_svm_synthetic_ub_step_',num2str(step),'_x1_',num2str(initial),'.txt'],'ub','-ascii')
% 
%% GS
rng(ra);
algo = 'gs';
i = 1;
sigma = f_sigma2;

% Setting constants c1 and c2
c1 = L*(d+3)^(3/2)/2;
c2 = L^2*(d+3)^(3)/2;
c3 = 2*(d+5)*(B^2 + sigma^2);


 disp('--------------------------------Starting GS----------------------------------');
for N = NList
    str=sprintf('N = %d', N); disp(str);
    
    % Setting step size, Probability mass function and eta
    eta_0 = (d)^(-1);   
    eta = eta_0/(N^(1/2)).*ones(N,1);
    c = sqrt((2*(f_x_1)*N)/(L*(c1^2*eta_0^2 + eta_0^2*c2 + N*c3)));
 
   
    if step == 0
       gamma = ones(N,1)*min((2*C_l + 2*c_11*eta(1)^2 - 1)/L, c/(N^(2/3)));
    elseif step == 1
       gamma = ones(N,1)*min(1/L, c/(N^(1/2)));
    elseif step == 2
       gamma = ones(N,1).*1/L;
    elseif step == 3
       gamma = 1/L.*1./(1:N); % Decreasing step size 
       eta = ones(N,1);
    end
    P_R = gamma./sum(gamma);
    
   gs(i,2:5) = rsqn_svm(algo, N, T, approx_samples, x_1, gamma, P_R, eta, lambda, training_u, training_v, test_u, test_v);
    i = i+1;
end
disp(gs(:,[2,4]));
 disp('--------------------------------End GS----------------------------------');

plot(NList,gs(:,2));
save(['RSQN_svm_synthetic_gs_step_',num2str(step),'_x1_',num2str(initial),'.txt'],'gs','-ascii')

%% 1SPSA
rng(ra);
algo = 'spsa';
i = 1;
sigma = f_sigma2;

% Setting constants c1 and c2
c1 = alpha_0*d^3/6;
c2 = 2*(alpha_1 + sigma^2)*d/4; 

 disp('--------------------------------Starting 1SPSA----------------------------------');
for N = NList
    str=sprintf('N = %d', N); disp(str);
    
    % Setting step size, Probability mass function and eta
    eta_0 = d^(-5/6);
    eta = eta_0/(N^(1/6)).*ones(N,1);
    c = sqrt((2*(f_x_1)*N*eta_0^2)/(L*c1^2*eta_0^6 + N*L*c2));
    
    if step == 0
       gamma = ones(N,1)*min((2*C_l - 1)/L, 1/((d^2*N)^(2/3)));
    elseif step == 1
       gamma = ones(N,1)*min((2*C_l + 2*c_11*eta(1)^2 - 1)/L, c/(N^(2/3)));
    elseif step == 2
       gamma = ones(N,1).*(2*C_l + 2*c_11*eta(1)^2 - 1)/L;
    elseif step == 3
       gamma = 1/L.*1./(1:N); % Decreasing step size 
       eta = ones(N,1);
    end
    
    P_R = gamma./sum(gamma);
    
    spsa(i,2:5) = rsqn_svm(algo, N, T, approx_samples, x_1, gamma, P_R, eta, lambda, training_u, training_v, test_u, test_v);
    i = i+1;
end
disp(spsa(:,[2,4]));
disp('--------------------------------End 1SPSA----------------------------------');

plot(NList,spsa(:,2));
save(['RSQN_svm_synthetic_spsa_step_',num2str(step),'_x1_',num2str(initial),'.txt'],'spsa','-ascii')

%% 1RDSA_Uniform
rng(ra);
algo = 'rdsa_u';
i = 1;

u = 1; % Setting 1RDSA_Uniform distribution parameter

% Setting constants c1 and c2
c1 = alpha_0*d^3*u^4/(6*1);
c2 = (alpha_1 + sigma^2)*(u)^2*d*2/1; 
 disp('--------------------------------Starting 1RDSA_Uniform----------------------------------');
for N = NList
    str=sprintf('N = %d \n', N); disp(str);
    
    % Setting step size, Probability mass function and eta
    eta_0 = d^(-5/6);
    eta = eta_0/(N^(1/6)).*ones(N,1);
    c = sqrt((2*(f_x_1)*N*eta_0^2)/(L*c1^2*eta_0^6 + N*L*c2));
    
    if step == 0
       gamma = ones(N,1)*min((2*C_l - 1)/L, 1/((d^2*N)^(2/3)));
    elseif step == 1
       gamma = ones(N,1)*min((2*C_l + 2*c_11*eta(1)^2 - 1)/L, c/(N^(2/3)));
    elseif step == 2
       gamma = ones(N,1).*(2*C_l + 2*c_11*eta(1)^2 - 1)/L;
    elseif step == 3
       gamma = 1/L.*1./(1:N); % Decreasing step size 
       eta = ones(N,1);
    end
    
    P_R = gamma./sum(gamma);
    
    rdsa_u(i,2:5) = rsqn_svm(algo, N, T, approx_samples, x_1, gamma, P_R, eta, lambda, training_u, training_v, test_u, test_v);
    i = i+1;
end
disp(rdsa_u(:,[2,4]));
disp('--------------------------------End 1RDSA_Uniform----------------------------------');

plot(NList,rdsa_u(:,2));
save(['RSQN_svm_synthetic_rdsa_u_step_',num2str(step),'_x1_',num2str(initial),'.txt'],'rdsa_u','-ascii')

% 1RDSA_Asymber
rng(ra);
algo = 'rdsa_ab';
i = 1;

ab = 0.0001; % Setting 1RDSA_Asymber distribution parameter

% Setting constants c1 and c2
c1 = alpha_0*d^3*(1 + ab)^4/6;
c2 = (alpha_1 + sigma^2)*(1 + ab)*d/2;

 disp('--------------------------------Starting 1RDSA_Asymber----------------------------------');
for N = NList+1
    str=sprintf('N = %d \n', N); disp(str);
    
    % Setting step size, Probability mass function and eta
    eta_0 = d^(-5/6);
    eta = eta_0/(N^(1/6)).*ones(N,1);
    c = sqrt((2*(f_x_1)*N*eta_0^2)/(L*c1^2*eta_0^6 + N*L*c2));
    
    if step == 0
       gamma = ones(N,1)*min((2*C_l - 1)/L, 1/((d^2*N)^(2/3)));
    elseif step == 1
       gamma = ones(N,1)*min((2*C_l + 2*c_11*eta(1)^2 - 1)/L, c/(N^(2/3)));
    elseif step == 2
       gamma = ones(N,1).*(2*C_l + 2*c_11*eta(1)^2 - 1)/L;
    elseif step == 3
       gamma = 1/L.*1./(1:N); % Decreasing step size 
       eta = ones(N,1);
    end
    
    P_R = gamma./sum(gamma);
    
    rdsa_ab(i,2:5) = rsqn_svm(algo, N, T, approx_samples, x_1, gamma, P_R, eta, lambda, training_u, training_v, test_u, test_v);
    i = i+1;
end
 disp(rdsa_ab(:,[2,4]));
 disp(' -------------------------------End 1RDSA_Asymber----------------------------------');
 
plot(NList,rdsa_ab(:,2));
save(['RSQN_svm_synthetic_rdsa_ab_step_',num2str(step),'_x1_',num2str(initial),'.txt'],'rdsa_ab','-ascii')

%% 1RDSA_Perm_DP
algo = 'rdsa_perm';
i = 1;

% Setting constants c1 and c2
c1 = alpha_0*d^3/6;
c2 = (alpha_1 + sigma^2)*d/2;
 disp('--------------------------------Starting 2RDSA_Perm_DP----------------------------------');
for N = NList
    str=sprintf('N = %d \n', N); disp(str);
    
    % Setting step size, Probability mass function and eta
    eta_0 = d^(-5/6);
    eta = eta_0/(N^(1/6)).*ones(N,1);
    c = sqrt((2*(f_x_1)*N*eta_0^2)/(L*c1^2*eta_0^6 + N*L*c2));
    
    if step == 0
       gamma = ones(N,1)*min((2*C_l - 1)/L, 1/((d^2*N)^(2/3)));
    elseif step == 1
       gamma = ones(N,1)*min((2*C_l + 2*c_11*eta(1)^2 - 1)/L, c/(N^(2/3)));
    elseif step == 2
       gamma = ones(N,1).*(2*C_l + 2*c_11*eta(1)^2 - 1)/L;
    elseif step == 3
       gamma = 1/L.*1./(1:N); % Decreasing step size 
       eta = ones(N,1);
    end
    
    P_R = gamma./sum(gamma);
    
    rdsa_perm(i,2:5) = rsqn_svm(algo, N, T, approx_samples, x_1, gamma, P_R, eta, lambda, training_u, training_v, test_u, test_v);
    i = i+1;
end
 disp(rdsa_perm(:,[2,4]));
 disp('--------------------------------End 2RDSA_Perm_DP----------------------------------');
plot(NList,rdsa_perm(:,2));
save(['RSQN_svm_synthetic_rdsa_perm_step_',num2str(step),'_x1_',num2str(initial),'.txt'],'rdsa_perm','-ascii')

%% 1RDSA_Lex_DP
% rng(ra);
% algo = 'rdsa_lex';
% i = 1;
% 
% % Setting constants c1 and c2
% c1 = alpha_0*d^3*3^(d-1);
% c2 = (alpha_1 + sigma^2)*d*2;
%  disp('--------------------------------Starting 1RDSA_Lex_DP----------------------------------');
% for N = NList
%     str=sprintf('N = %d \n', N); disp(str);
%     
%     % Setting step size, Probability mass function and eta
%     eta_0 = d^(-5/6);
%     eta = eta_0/(N^(1/6)).*ones(N,1);
%     c = sqrt((2*(f_x_1)*N*eta_0^2)/(L*c1^2*eta_0^6 + N*L*c2));
%     
%     if step == 0
%        gamma = ones(N,1)*min((2*C_l - 1)/L, 1/((d^2*N)^(2/3)));
%     elseif step == 1
%        gamma = ones(N,1)*min((2*C_l + 2*c_11*eta(1)^2 - 1)/L, c/(N^(2/3)));
%     elseif step == 2
%        gamma = ones(N,1).*(2*C_l + 2*c_11*eta(1)^2 - 1)/L;
%     elseif step == 3
%        gamma = 1/L.*1./(1:N); % Decreasing step size 
%        eta = ones(N,1);
%     end
%     P_R = gamma./sum(gamma);
%     
%     rdsa_lex(i,2:5) = rsqn_svm(algo, N, T, approx_samples, x_1, gamma, P_R, eta, lambda, training_u, training_v, test_u, test_v);
%     i = i+1;
% end
%  disp(rdsa_lex(:,[2,4]));
%  disp('--------------------------------End 1RDSA_Lex_DP----------------------------------');
% plot(NList,rdsa_lex(:,2));
% save(['RSQN_svm_synthetic_rdsa_lex_step_',num2str(step),'_x1_',num2str(initial),'.txt'],'rdsa_kw','-ascii')


%% Plotting results

figure;
plot(NList,ub(:,2));
hold on
plot(NList,gs(:,2));
plot(NList,spsa(:,2));
plot(NList,rdsa_u(:,2));
plot(NList,rdsa_ab(:,2));
% plot(NList,rdsa_lex(:,2));
plot(NList,rdsa_perm(:,2));
title('SNG');
% legend('Unbiased','GS','SPSA','RDSA-Unif','RDSA-AsymBer','RDSA-Lex-DP','RDSA-Perm-DP');
legend('Unbiased','GS','SPSA','RDSA-Unif','RDSA-AsymBer','RDSA-Perm-DP');

% 
% %% Saving results to txt file
% 
save(['RSQN_svm_synthetic_step_',num2str(step),'_x1_',num2str(initial)]);
