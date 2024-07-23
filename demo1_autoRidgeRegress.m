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
%         k ~ N(0,1/alpha * I)      % prior on weights
%  y | x, k ~ N(x^T k, nsevar) % linear-Gaussian observations
%
% Empirical Bayes inference:
%
% 1. [alpha_hat, nsevar_hat] = arg max P(Y | X, alpha,nsevar);  
% 2.     k_hat = arg max_k  P(k | Y, X, alpha_hat)

% set path
addpath tools
addpath inference/

%% 1. Make a simulated dataset

nk = 100;     % number of regression coefficients 
nsamps = 200; % number of samples
signse = 5;   % stdev of added noise

% make filter
tt = (1:nk)'; % coefficient indices
k = gsmooth(randn(nk,1),3); % generate smooth weight vector

% make design matrix
Xdsgn = randn(nsamps,nk);

% simulate outputs
y = Xdsgn*k + randn(nsamps,1)*signse; 

%% 2.  Compute ML and ridge regression estimates

% Compute sufficient statistics
dd.x = Xdsgn;
dd.y = y;
dd.xx = Xdsgn'*Xdsgn;   
dd.xy = Xdsgn'*y;
dd.yy = y'*y;
dd.nx = nk;
dd.ny = nsamps;

% real data
% dd.xx = kron(eye(size(Yout,2)),XX);
% dd.xy = vec(XY);
% dd.yy = sum(sum(Yout.^2));
% dd.nx = numel(dd.xy);
% dd.ny = prod(size(Yout));

% Compute ML estimate
kml = dd.xx\dd.xy;  

% Compute EB ridge regression estimate
alpha0 = 10; % initial guess at alpha
[k_vi,hprs_vi] = autoRidgeRegress_VI(dd);
[k_vi2,hprs_vi2] = autoRidgeRegress_VI(dd);
% kml = k_vi;
[kridge,hprs_hat] = autoRidgeRegress_fixedpoint(dd,alpha0);
hprs_vi
hprs_hat 

[w, V, invV, logdetV, an, bn, E_a, L] = vb_linear_fit(Xdsgn, y);

% ax = np; plt_cmp_vectors([kml k_vi kridge],[],'lines',ax);ef;
% plt_cmp_vectors([kml k_vi kridge w],{'ml','vi','fp','vid'});ef;
% plt_cmp_vectors([k_vi k_vi2 kridge w],{'vi1','vi2','fp','vid'},'hist');ef;
plt_cmp_vectors([k_vi kridge],{'vi','fixed point'},'lines');ef;

return

% [kridge,hprs_hat] = autoRidgeRegress_fixedpoint(dd,alpha0);
% autoRidgeRegress: jcount=2, alpha=407.356, nsevar=1.075, dparams=204675.069143
% autoRidgeRegress: jcount=3, alpha=334.318, nsevar=1.059, dparams=73.038710
% autoRidgeRegress: jcount=4, alpha=317.117, nsevar=1.057, dparams=17.200295
% autoRidgeRegress: jcount=5, alpha=312.791, nsevar=1.057, dparams=4.326219
% autoRidgeRegress: jcount=6, alpha=311.682, nsevar=1.056, dparams=1.108718
% autoRidgeRegress: jcount=7, alpha=311.397, nsevar=1.056, dparams=0.285543
% autoRidgeRegress: jcount=8, alpha=311.323, nsevar=1.056, dparams=0.073633
% autoRidgeRegress: jcount=9, alpha=311.304, nsevar=1.056, dparams=0.018994
% autoRidgeRegress: jcount=10, alpha=311.299, nsevar=1.056, dparams=0.004900
% autoRidgeRegress: jcount=11, alpha=311.298, nsevar=1.056, dparams=0.001264
% autoRidgeRegress: jcount=12, alpha=311.298, nsevar=1.056, dparams=0.000326
% autoRidgeRegress: jcount=13, alpha=311.298, nsevar=1.056, dparams=0.000084
% autoRidgeRegress: jcount=14, alpha=311.298, nsevar=1.056, dparams=0.000022
% autoRidgeRegress: jcount=15, alpha=311.298, nsevar=1.056, dparams=0.000006
% autoRidgeRegress: jcount=16, alpha=311.298, nsevar=1.056, dparams=0.000001
% autoRidgeRegress: jcount=17, alpha=311.298, nsevar=1.056, dparams=0.000000
% Finished autoRidgeRegression in #17 steps

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
