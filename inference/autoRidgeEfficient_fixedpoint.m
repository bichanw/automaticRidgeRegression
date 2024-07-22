function [khat,hprs] = autoRidgeEfficient_fixedpoint(X,Y,lam0,opts)
% "Automatic" ridge regression w/ fixed-point evidence optimization for hyperparams 
%  
% [khat,hprs] = autoRidgeRegression_fp(datastruct,lam0,opts)
%
% Computes maximum marginal likelihood estimate for prior variance of an
% isotropic Gaussian prior (1/alpha) and variance of additive noise (nsevar)
%
% Note: traditional ridge parameter equals [prior precision] * [noise variance], 
%       i.e., lambda = (hprs.alpha * hprs.nsevar)
%
% INPUT:
% -------
%   X - stimulus matrix (T x nx)
%   Y - response matrix (T x ny)
%   lam0 - initial value of the ratio of ridge parameter (OPTIONAL)
%            (equal to noise variance divided by prior variance)
%   opts - options stucture w fields  'maxiter' and 'tol' (OPTIONAL)
%
%
% OUTPUT:
% -------
%     khat - "empirical bayes' estimate of kernel k
%     hprs - struct with fitted hyperparameters 'alpha' and 'nsevar'
%
%  Updated 2015.03.24 (jwp)

MAXALPHA = 1e6; % Maximum allowed value for prior precision

% Check input arguments
if nargin < 2
    lam0 = 10;
end
if nargin < 3
    opts.maxiter = 100;
    opts.tol = 1e-6;
end

% ----- Initialize some stuff -------
jcount = 1;  % counter
dparams = inf;  % Change in params from previous step

% extract sufficient statistics
xy = X'*Y;
xx = X'*X;
yy = Y'*Y;
% dimensions of data
nx = size(X,2);
ny = size(Y,2);
T = size(X,1);
d = nx * ny;
N = T * ny;


Lmat = speye(nx);  % Diagonal matrix for prior

% ------ Initialize alpha & nsevar using MAP estimate around lam0 ------
kmap0  = (xx + lam0*Lmat)\xy;  % MAP estimate given lam0
nsevar = sum(sum((Y - X*kmap0).^2)); % 1st estimate for nsevar: var(y-x*kmap0); 
alpha  = lam0/nsevar;
Y_norm = sum(sum(Y.^2));

% ------ Run fixed-point algorithm  ------------
while (jcount <= opts.maxiter) && (dparams>opts.tol) && (alpha <= MAXALPHA)
    
    Cpost_small = inv(xx/nsevar + Lmat*alpha);  % posterior covariance before taking kronecker with I_ny

    
    mupost = vec(Cpost_small / nsevar * xy); % posterior mean
    alphanew = (d - alpha*ny.*trace(Cpost_small))./sum(mupost.^2); % update for alpha
    
    numerator = Y_norm - 2*mupost'*vec(xy) + mupost'*vec(xx * reshape(mupost,nx,ny));
    nsevarnew = sum(numerator)./(N - d + alpha * ny * trace(Cpost_small));

    % update counter, alpha & nsevar
    dparams = norm([alphanew;nsevarnew]-[alpha;nsevar]);
    jcount = jcount+1;
    alpha = alphanew;
    nsevar = nsevarnew;

    fprintf('autoRidgeRegress: jcount=%d, alpha=%.3f, nsevar=%.3f, dparams=%f\n', ...
        jcount, alpha, nsevar, dparams);
end
khat = (xx + alpha*nsevar*Lmat)\(xy); % final estimate (posterior mean after last update)

if alpha >= MAXALPHA
    fprintf(1, 'Finished autoRidgeRegression: filter is all-zeros\n');
    khat = mupost*0;  % Prior variance is delta function
elseif jcount < opts.maxiter
    fprintf('Finished autoRidgeRegression in #%d steps\n', jcount);
else
    fprintf(1, 'Stopped autoRidgeRegression: MAXITER (%d) steps; dparams last step: %f\n', ...
        jcount, dparams);
end

% Put hyperparameter estimates into struct
hprs.alpha  = alpha;
hprs.nsevar = nsevar;
hprs.lambda = alpha * nsevar;

end