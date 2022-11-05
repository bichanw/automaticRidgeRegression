% demo1_autoRidgeRegress.m 
%
% Demo script to illustrate empirical Bayes (EB) ridge regression
%
% This is a two-step inference procedure: 
% 1. Use evidence optimization (aka "maximum marginal likelihood", aka
% "type-2 maximum likelihood") to infer ridge parameter
% 2. Find MAP estimate for weights in linear regression model given ridge
% prior
% 
% Model
% -----
%         k ~ N(0,1/alpha)      % prior on weights
%  y | x, k ~ N(x^T k, sigma^2) % linear-Gaussian observations
%
% Empirical Bayes inference:
%
% 1. alpha_hat = arg max_alpha P(Y | X, alpha);  
% 2.     k_hat = arg max_k     P(k | Y, X, alpha_hat)

% set path
addpath tools
addpath inference/

%% 1. Make a simulated dataset

nk = 100;     % number of regression coefficients 
nsamps = 200; % number of samples
signse = 3;   % stdev of added noise

% make filter
tt = (1:nk)'; % coefficient indices
k = gsmooth(randn(nk,1),3); % generate smooth weight vector

% make design matrix
Xdsgn = randn(nsamps,nk);

% simulate outputs
y = Xdsgn*k + randn(nsamps,1)*signse; 

%% 2.  Compute ML and ridge regression estimates

% Compute sufficient statistics
dd.xx = Xdsgn'*Xdsgn;   
dd.xy = Xdsgn'*y;
dd.yy = y'*y;
dd.nx = nk;
dd.ny = nsamps;

% Compute ML estimate
kml = dd.xx\dd.xy;  

% Compute EB ridge regression estimate
alpha0 = 1; % initial guess at alpha
[kridge,hprs_hat] = autoRidgeRegress_fixedpoint(dd,alpha0);


%% 3. Compare performance & make plots

clf; 
plot(tt, k,'k',tt, kml, tt, kridge);
legend('true k', 'ML', 'ridge');

fprintf('\nInferred hyperparameters:\n');
fprintf('-----------------------\n');
fprintf('alpha  = %.2f\n',hprs_hat.alpha);
fprintf('nsevar = %.2f (true = %.2f)\n\n',hprs_hat.nsevar, signse^2);

% Compare errors
r2fun = @(kest)(1-sum((k-kest).^2)/sum(k.^2));
fprintf('Performance comparison:\n');
fprintf('-----------------------\n');
fprintf('   ML: R2 = %.3f\n', r2fun(kml));
fprintf('ridge: R2 = %.3f\n', r2fun(kridge));
