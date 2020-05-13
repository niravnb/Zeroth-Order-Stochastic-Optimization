close all; clear; clc; 
%% Input Parameters
ra = rng(105,'twister'); % Setting some random seed for reproducibility
step = 0; % Setting type of stepsize; value of 0 mean min{1/L, 1/(d^2N)^2/3}, 1 means min{1/L, c/sqrt(N)}, 2 means 1\L, 3 means 1\LK with eta = 1

d = 5; % Dimension of the problem
initial = 5; % Initial point
x_1 = initial.*ones(d,1); % Initial point of d dimension
x_star = 10*ones(d,1); % Optimal point
type = 1; % 1 = Multimodal function
f_x_star = SZO(d, x_star, 0, type); % Getoptimum function value 
f_x_1 = SZO(d, x_1, 0, type); % Get function value at initial point

NList = [50, 500, 1000, 1850, 3250, 5000]; % Iterations list
T = 50; % Replications
Q = 10000; % To find quality of solution at x_R

%% Declaring variables to store results

ub = [NList', zeros(length(NList),6)]; % Itr, mean_sng, std. err sng, mean f-diff, std. err f-diff, mean NMSE, std. err NMSE
gs = [NList', zeros(length(NList),6)];
spsa = [NList', zeros(length(NList),6)];

rdsa_u = [NList', zeros(length(NList),6)];
rdsa_ab = [NList', zeros(length(NList),6)];

rdsa_lex = [NList', zeros(length(NList),6)];
rdsa_perm = [NList', zeros(length(NList),6)];
rdsa_kw = [NList', zeros(length(NList),6)];

%% Paramater Estimation L and sigma
N_0 = 300; % Using N_0 i.i.d. samples 
s = 0.3; %0.3; 

a = 0; % Lowest value of x 
b = 100; % Highest value of x

gradient_var = zeros(N_0,1);
B_list = zeros(N_0,1);
hessian_norm = zeros(N_0,1);
third_der_mag = zeros(N_0,1);

for i = 1:N_0 % loop over N_0 random points x
    x = a + (b-a).*rand(d,1);
    
    hessian = get_hessian(d, x, type); % Getting hessian
    hessian_norm(i) = norm(hessian,2);
    
    gradient_norm = zeros(N_0,1);
    grad_l2norm = 0;
    for j = 1:N_0 % loop over N_0 i.i.d. samples
       gradient = SFO(d, x, s, type, 'ub', 0);  % calling SFO
       gradient_norm(j) = norm(gradient,2)^2;
       grad_l2norm = grad_l2norm + norm(gradient,2);
    end
    gradient_var(i) = var(gradient_norm);
    B_list(i) = grad_l2norm/N_0;
    
    temp = multimodal_third_derivative(x);
    third_der_mag(i) = max(abs(temp));
end

L = max(hessian_norm); % Estimation of L
sigma = max(gradient_var); % Estimation of sigma
B = max(B_list); % Estimation of B

alpha_0 = max(third_der_mag); % Estimation of third derivative upper bound
alpha_1 = (1*d)^2; % Estimation of E[f(x + eta*delta)^2] upper bound

%% RSG_Unbiased algorithm

rng(ra);
algo = 'ub';
i = 1;
 disp('--------------------------------Starting Unbiased----------------------------------');
for N = NList
    str=sprintf('N = %d \n', N); disp(str);

    % Setting step size and Probability mass function
    eta = 0*ones(N,1);
    c = sqrt(2*(f_x_1-f_x_star)/(L*sigma));
    
    if step == 0
       gamma = ones(N,1)*min(1/L, c/(sqrt(N)));
    elseif step == 1
       gamma = ones(N,1)*min(1/L, c/(sqrt(N)));
    elseif step == 2
       gamma = ones(N,1).*1/L;
    elseif step == 3
       gamma = 1/L.*1./(1:N); % Decreasing step size 
       eta = ones(N,1);
    end
    
    P_R = gamma./sum(gamma);
    
    ub(i,2:7) = rsg(d, type, algo, N, T, Q, x_1, x_star, gamma, P_R, eta, sigma);
    i = i+1;
end

disp(ub(:,2:3));
 disp('--------------------------------End Unbiased----------------------------------');
 
plot(NList,ub(:,2));
save(['multimodal_ub_dim_',num2str(d),'_x1_',num2str(initial),'.txt'],'ub','-ascii')

%% GS
rng(ra);
algo = 'gs';
i = 1;
sigma = 0.03;

% Setting constants c1 and c2
c1 = L*(d+3)^(3/2)/2;
c2 = L^2*(d+3)^(3)/2;
c3 = 2*(d+5)*(B^2 + sigma^2);

% step = 1;

 disp('--------------------------------Starting GS----------------------------------');
for N = NList
    str=sprintf('N = %d', N); disp(str);
    
    % Setting step size, Probability mass function and eta
    eta_0 = sqrt(2*f_x_1/L);   
    eta = eta_0/(N^(1/2)).*ones(N,1);
    c = sqrt((2*(f_x_1)*N)/(L*(c1^2*eta_0^2 + eta_0^2*c2 + N*c3)));
  
    if step == 0
       gamma = ones(N,1)*min(1/L, c/(sqrt(N)));
    elseif step == 1
       gamma = ones(N,1)*min(1/L, c/(N^(1/2)));
    elseif step == 2
       gamma = ones(N,1).*1/L;
    elseif step == 3
       gamma = 1/L.*1./(1:N); % Decreasing step size 
       eta = ones(N,1);
    end
    P_R = gamma./sum(gamma);
    
    gs(i,2:7) = rsg(d, type, algo, N, T, Q, x_1, x_star, gamma, P_R, eta, sigma);
    i = i+1;
end
disp(gs(:,2:3));
 disp('--------------------------------End GS----------------------------------');

plot(NList,gs(:,2));
save(['multimodal_gs_dim_',num2str(d),'_x1_',num2str(initial),'.txt'],'gs','-ascii')


%% 1SPSA
rng(ra);
algo = 'spsa';
i = 1;

% Setting constants c1 and c2
c1 = alpha_0*d^3/6;
c2 = 2*(alpha_1 + sigma^2)*d/1;

% step = 2;

 disp('--------------------------------Starting 1SPSA----------------------------------');
for N = NList
    str=sprintf('N = %d', N); disp(str);
    
    % Setting step size, Probability mass function and eta
    eta_0 = d^(-5/6);
    eta = eta_0/(N^(1/6)).*ones(N,1);
    c = sqrt((2*(f_x_1)*N*eta_0^2)/(L*c1^2*eta_0^6 + N*L*c2));

   
    if step == 0
       gamma = ones(N,1)*min(1/L, 1/(d^2*N)^(2/3));
    elseif step == 1
       gamma = ones(N,1)*min(1/L, c/(N^(2/3)));
    elseif step == 2
       gamma = ones(N,1).*1/L;
    elseif step == 3
       gamma = 1/L.*1./(1:N); % Decreasing step size 
       eta = ones(N,1);
    end
    P_R = gamma./sum(gamma);
    
    spsa(i,2:7) = rsg(d, type, algo, N, T, Q, x_1, x_star, gamma, P_R, eta, sigma);
    i = i+1;
end
disp(spsa(:,2:3));
 disp('--------------------------------End 1SPSA----------------------------------');

plot(NList,spsa(:,2));
save(['multimodal_spsa_dim_',num2str(d),'_x1_',num2str(initial),'.txt'],'spsa','-ascii')

%% 1RDSA_Uniform
rng(ra);
algo = 'rdsa_u';
i = 1;
u = 1; % 1RDSA_Uniform distribution parameter

% Setting constants c1 and c2
c1 = alpha_0*d^3*u^4/(6*1);
c2 = (alpha_1 + sigma^2)*(u)^2*d*2/1; %alpha_1*d/2;

 disp('--------------------------------Starting 1RDSA_Uniform----------------------------------');
for N = NList
%     str=sprintf('N = %d \n', N); disp(str);
    
    % Setting step size, Probability mass function and eta
    eta_0 = d^(-5/6);
    eta = eta_0/(N^(1/6)).*ones(N,1);
    c = sqrt((2*(f_x_1-f_x_star)*N*eta_0^2)/(L*c1^2*eta_0^6 + N*L*c2));
    
    if step == 0
       gamma = ones(N,1)*min(1/L, 1/(d^2*N)^(2/3));
    elseif step == 1
       gamma = ones(N,1)*min(1/L, c/(N^(2/3)));
    elseif step == 2
       gamma = ones(N,1).*1/L;
    elseif step == 3
       gamma = 1/L.*1./(1:N); % Decreasing step size 
       eta = ones(N,1);
    end
    
    P_R = gamma./sum(gamma);
    
    rdsa_u(i,2:7) = rsg(d, type, algo, N, T, Q, x_1, x_star, gamma, P_R, eta, sigma);
    i = i+1;
end
disp(rdsa_u(:,2:3));
 disp('--------------------------------End 1RDSA_Uniform----------------------------------');
plot(NList,rdsa_u(:,2));
save(['multimodal_rdsa_u_dim_',num2str(d),'_x1_',num2str(initial),'.txt'],'rdsa_u','-ascii')

%% 1RDSA_Asymber
rng(ra);
algo = 'rdsa_ab';
i = 1;

ab = 0.0001; % 1RDSA_Asymber distribution parameter

% Setting constants c1 and c2
c1 = alpha_0*d^3*(1 + ab)^4/6;
c2 = (alpha_1 + sigma^2)*(1 + ab)*d*2;

disp('--------------------------------Starting 1RDSA_Asymber----------------------------------');
for N = NList+1
    str=sprintf('N = %d \n', N); disp(str);
    
   % Setting step size, Probability mass function and eta
    eta_0 = d^(-5/6);
    eta = eta_0/(N^(1/6)).*ones(N,1);
    c = sqrt((2*(f_x_1)*N*eta_0^2)/(L*c1^2*eta_0^6 + N*L*c2));
    
    if step == 0
       gamma = ones(N,1)*min(1/L, 1/(d^2*N)^(2/3));
    elseif step == 1
       gamma = ones(N,1)*min(1/L, c/(N^(2/3)));
    elseif step == 2
       gamma = ones(N,1).*1/L;
    elseif step == 3
       gamma = 1/L.*1./(1:N); % Decreasing step size 
       eta = ones(N,1);
    end
    
    P_R = gamma./sum(gamma);
    
    rdsa_ab(i,2:7) = rsg(d, type, algo, N, T, Q, x_1, x_star, gamma, P_R, eta, sigma);
    i = i+1;
end
 disp(rdsa_ab(:,2:3));
 disp(' -------------------------------End 1RDSA_Asymber----------------------------------');
plot(NList,rdsa_ab(:,2));
save(['multimodal_rdsa_ab_dim_',num2str(d),'_x1_',num2str(initial),'.txt'],'rdsa_ab','-ascii')



%% 1RDSA_Perm_DP
algo = 'rdsa_perm';
i = 1;

% Setting constants c1 and c2
c1 = alpha_0*d^3/6;
c2 = (alpha_1 + sigma^2)*d/2;
 disp('--------------------------------Starting 1RDSA_Perm_DP----------------------------------');
for N = NList
    str=sprintf('N = %d \n', N); disp(str);
    
    % Setting step size, Probability mass function and eta
    eta_0 = d^(-5/6);
    eta = eta_0/(N^(1/6)).*ones(N,1);
    c = sqrt((2*(f_x_1)*N*eta_0^2)/(L*c1^2*eta_0^6 + N*L*c2));

    if step == 0
       gamma = ones(N,1)*min(1/L, 1/(d^2*N)^(2/3));
    elseif step == 1
       gamma = ones(N,1)*min(1/L, c/(N^(2/3)));
    elseif step == 2
       gamma = ones(N,1).*1/L;
    elseif step == 3
       gamma = 1/L.*1./(1:N); % Decreasing step size 
       eta = ones(N,1);
    end
    
    P_R = gamma./sum(gamma);
    
    rdsa_perm(i,2:7) = rsg(d, type, algo, N, T, Q, x_1, x_star, gamma, P_R, eta, sigma);
    i = i+1;
end
 disp(rdsa_perm(:,2:3));
 disp('--------------------------------End 1RDSA_Perm_DP----------------------------------');
plot(NList,rdsa_perm(:,2));
save(['multimodal_rdsa_perm_dim_',num2str(d),'_x1_',num2str(initial),'.txt'],'rdsa_perm','-ascii')

 %% 1RDSA_KW_DP
rng(ra);
algo = 'rdsa_kw';
i = 1;

% Setting constants c1 and c2
c1 = alpha_0*d^3/6;
c2 = (alpha_1 + sigma^2)*d/2;

 disp('--------------------------------Starting 1RDSA_KW_DP----------------------------------');
for N = NList
    str=sprintf('N = %d \n', N); disp(str);
    
    % Setting step size, Probability mass function and eta
    eta_0 = d^(-5/6);
    eta = eta_0/(N^(1/6)).*ones(N,1);
    c = sqrt((2*(f_x_1-f_x_star)*N*eta_0^2)/(L*c1^2*eta_0^6 + N*L*c2));
    
    if step == 0
       gamma = ones(N,1)*min(1/L, 1/(d^2*N)^(2/3));
    elseif step == 1
       gamma = ones(N,1)*min(1/L, c/(N^(2/3)));
    elseif step == 2
       gamma = ones(N,1).*1/L;
    elseif step == 3
       gamma = 1/L.*1./(1:N); % Decreasing step size 
       eta = ones(N,1);
    end
    
    P_R = gamma./sum(gamma);
    
    rdsa_kw(i,2:7) = rsg(d, type, algo, N, T, Q, x_1, x_star, gamma, P_R, eta, sigma);
    i = i+1;
end
 disp(rdsa_kw(:,2:3));
 disp('--------------------------------End 1RDSA_KW_DP----------------------------------');
plot(NList,rdsa_kw(:,2));
save(['multimodal_rdsa_kw_dim_',num2str(d),'_x1_',num2str(initial),'.txt'],'rdsa_kw','-ascii')

%% 1RDSA_Lex_DP
% rng(ra);
% algo = 'rdsa_lex';
% i = 1;
% 
% % Setting constants c1 and c2
% c1 = alpha_0*d^3*3^(d-1);
% c2 = (alpha_1 + sigma^2)*d*2;
% disp('--------------------------------Starting 1RDSA_Lex_DP----------------------------------');
% 
% for N = NList
%     str=sprintf('N = %d \n', N); disp(str);
%     
%     % Setting step size, Probability mass function and eta
%     eta_0 = d^(-5/6);
%     eta = eta_0/(N^(1/6)).*ones(N,1);
%     c = sqrt((2*(f_x_1-f_x_star)*N*eta_0^2)/(L*c1^2*eta_0^6 + N*L*c2));
%     
%     if step == 0
%        gamma = ones(N,1)*min(1/L, 1/(d^2*N)^(2/3));
%     elseif step == 1
%        gamma = ones(N,1)*min(1/L, c/(N^(2/3)));
%     elseif step == 2
%        gamma = ones(N,1).*1/L;
%     elseif step == 3
%        gamma = 1/L.*1./(1:N); % Decreasing step size 
%        eta = ones(N,1);
%     end
%     
%     P_R = gamma./sum(gamma);
%     
%     rdsa_lex(i,2:7) = rsg(d, type, algo, N, T, Q, x_1, x_star, gamma, P_R, eta, sigma);
%     i = i+1;
% end
%  disp(rdsa_lex(:,2));
%  disp('--------------------------------End 1RDSA_Lex_DP----------------------------------');
% plot(NList,rdsa_lex(:,2));
% save(['multimodal_rdsa_lex_dim_',num2str(d),'_x1_',num2str(initial),'.txt'],'rdsa_lex','-ascii')


%% Plotting results

% Plotting SNG
figure;
plot(NList,ub(:,2));
hold on
plot(NList,gs(:,2));
plot(NList,spsa(:,2));
plot(NList,rdsa_u(:,2));
plot(NList,rdsa_ab(:,2));
% plot(NList,rdsa_lex(:,2));
plot(NList,rdsa_perm(:,2));
plot(NList,rdsa_kw(:,2));
title('SNG');
% legend('Unbiased','GS','SPSA','RDSA-Unif','RDSA-AsymBer','RDSA-Lex-DP','RDSA-Perm-DP','RDSA-KW-DP');
legend('Unbiased','GS','SPSA','RDSA-Unif','RDSA-AsymBer','RDSA-Perm-DP','RDSA-KW-DP');



%%

% Plotting F-diff
figure;
plot(NList,ub(:,4));
hold on
plot(NList,gs(:,4));
plot(NList,spsa(:,4));
plot(NList,rdsa_u(:,4));
plot(NList,rdsa_ab(:,4));
% plot(NList,rdsa_lex(:,4));
plot(NList,rdsa_perm(:,4));
plot(NList,rdsa_kw(:,4));
title('f-diff');
% legend('Unbiased','GS','SPSA','RDSA-Unif','RDSA-AsymBer','RDSA-Lex-DP','RDSA-Perm-DP','RDSA-KW-DP');
legend('Unbiased','GS','SPSA','RDSA-Unif','RDSA-AsymBer','RDSA-Perm-DP','RDSA-KW-DP');


% Plotting NMSE
figure;
plot(NList,ub(:,6));
hold on
plot(NList,gs(:,6));
plot(NList,spsa(:,6));
plot(NList,rdsa_u(:,6));
plot(NList,rdsa_ab(:,6));
% plot(NList,rdsa_lex(:,6));
plot(NList,rdsa_perm(:,6));
plot(NList,rdsa_kw(:,6));
title('NMSE');
% legend('Unbiased','GS','SPSA','RDSA-Unif','RDSA-AsymBer','RDSA-Lex-DP','RDSA-Perm-DP','RDSA-KW-DP');
legend('Unbiased','GS','SPSA','RDSA-Unif','RDSA-AsymBer','RDSA-Perm-DP','RDSA-KW-DP');




%% Saving results to txt file

save(['rsg_multimodal_dim_',num2str(d),'_x1_',num2str(initial)]);
