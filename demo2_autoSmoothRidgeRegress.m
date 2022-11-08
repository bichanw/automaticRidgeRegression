% demo2_autoCorrRidgeRegress.m 
%
% Demo script to illustrate empirical Bayes (EB) inference for
% linear-Gaussian regression model with "smooth ridge" prior. 
% 
% This prior corresponds to a Gaussian process prior with an exponential
% covariance function.  The tridiagonal inverse covariance matrix is:
%
% C^-1 = (1/alpha)[ 1     -rho
%                   -rho  1 + rho^2 -rho
%                          ....
%
%                          -rho  1+rho^2 -rho
%                                  -rho     1]
% 
% The covariance matrix has exponential falloff with shape rho^(-|dt|):
%
%  C(dt) = (1/alpha)*rho^(-|dt|) = (1/alpha) exp(dt*log(|dt|)).
%
% Model
% -----
%         k ~ N(0,C)      % prior on weights
%  y | x, k ~ N(x^T k, nsevar) % linear-Gaussian observations
%
% where inverse of the prior covariance C is given above. T

% set path
addpath tools
addpath inference/


%% 1. Make a simulated dataset

nk = 100;     % number of regression coefficients in filter
nsamps = 200; % number of samples
signse = 3;   % stdev of added noise

% Set up filter
k = zeros(nk,1);  % initialize stimulus filter
strtInds = (1:25:nk)';  % indices where filter has discontinuous jumps

% Create filter
for jj = 1:length(strtInds)-1
    inds = strtInds(jj):strtInds(jj+1)-1;
    k(inds) = gsmooth(randn(length(inds),1),3);
end
inds = strtInds(end):nk;
k(inds) = gsmooth(randn(length(inds),1),3); 

% make design matrix
Xdsgn = randn(nsamps,nk);

% simulate outputs
y = Xdsgn*k + randn(nsamps,1)*signse; 

%% 2. Compute ML and ridge regression estimates

% Compute sufficient statistics
dd.xx = Xdsgn'*Xdsgn;
dd.xy = Xdsgn'*y;
dd.yy = y'*y;
dd.ny = nsamps;

% maximum-likelihood estimate
kml = dd.xx\dd.xy;  

% Compute EB ridge regression estimate
alpha0 = 1; % initial guess at alpha
[kridge,hprs_ridge] = autoRidgeRegress_fixedpoint(dd,alpha0);


%% 3. Compute smooth ridge regression estimates (with or without breaks)

% Automatic smooth-ridge
[ksm1,hprs_sm1,Cinv_sm1] = autoSmoothRidgeRegress(dd);

% Automatic smooth-ridge, with information about breaks
[ksm2,hprs_sm2,Cinv_sm2] = autoSmoothRidgeRegress(dd,strtInds);

%% 4. Display results and make plots

subplot(221);
imagesc(inv(Cinv_sm1)); title('smooth prior cov');
xlabel('coeff #'); ylabel('coeff #');
subplot(222);
imagesc(inv(Cinv_sm2)); title('smooth prior cov w/ breaks');
xlabel('coeff #'); ylabel('coeff #');

subplot(224)
plot(tt, k,'k--',tt, kridge,tt,ksm1,tt,ksm2);
legend('true k','ridge', 'sm1', 'sm2');
xlabel('coeff #')
ylabel('coeff')

fprintf('\nInferred hyperparameters:\n');
fprintf('==========================\n');
fprintf('Ridge:\n')
fprintf('------\n');
fprintf('alpha  = %.2f\n',hprs_ridge.alpha);
fprintf('nsevar = %.2f (true = %.2f)\n',hprs_ridge.nsevar, signse^2);
fprintf('\nSmooth-ridge:\n')
fprintf('------------\n');
fprintf('alpha  = %.2f\n',hprs_sm1.alpha);
fprintf('nsevar = %.2f (true = %.2f)\n',hprs_sm1.nsevar, signse^2);
fprintf('   rho = %.2f\n',hprs_sm1.rho);
fprintf('\nSmooth-ridge w/ breaks:\n')
fprintf('----------------------\n');
fprintf('alpha  = %.2f\n',hprs_sm2.alpha);
fprintf('nsevar = %.2f (true = %.2f)\n',hprs_sm2.nsevar, signse^2);
fprintf('   rho = %.2f\n',hprs_sm2.rho);

% Compare errors
r2fun = @(kest)(1-sum((k-kest).^2)/sum(k.^2));
fprintf('\nPerformance comparison:\n');
fprintf('======================\n');
fprintf('       ML: R2 = %.3f\n', r2fun(kml));
fprintf('    ridge: R2 = %.3f\n', r2fun(kridge));
fprintf('   smooth: R2 = %.3f\n', r2fun(ksm1));
fprintf('w/ breaks: R2 = %.3f\n', r2fun(ksm2));

