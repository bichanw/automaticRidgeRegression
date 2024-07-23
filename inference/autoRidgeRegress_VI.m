function [khat,hprs] = autoRidgeRegress_VI(dstruct,opts)
% "Automatic" ridge regression w/ variational inference for hyperparams 
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
%  dstruct - data structure with fields:
%            .xx - stimulus autocovariance matrix X'*X
%            .xy - stimulus-response cross-covariance X'*Y
%            .yy - response variance Y'*Y
%            .ny - number of samples 
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
    opts.maxiter = 100;
    opts.tol = 1e-6;
end

% ----- Initialize some stuff -------
jcount = 1;  % counter
dparams = inf;  % Change in params from previous step

% extract sufficient statistics
x  = dstruct.x;
y  = dstruct.y;
xx = dstruct.xx; 
xy = dstruct.xy;
yy = dstruct.yy;
ny = dstruct.ny;

nx= size(xx,1); % number of stimulus dimnesions
Lmat = speye(nx);  % Diagonal matrix for prior
N = ny; % number of samples
d = dstruct.nx;


% initialize hyperparameters struct
% using non-informative gamma prior
ELBO_no_const = -inf; % Evidence lower bound
d_ELBO = inf;
c_0 = .001;
d_0 = .001;
d_vi = .001;
a_0 = .001;
b_0 = .001;



% VI parameters not affected by interations
c_vi = c_0 + d / 2;
a_vi = a_0 + N / 2;


to_save.ELBO = [];
% ------ Run variational inference algorithm  ------------
while (jcount <= opts.maxiter) && (abs(d_ELBO) >= opts.tol)
    
    % update q(beta, sigma2)
    inv_V_vi = (c_vi / d_vi) * speye(d) + xx;
    V_vi = inv(inv_V_vi);
    beta_vi = inv_V_vi \ xy;
    b_vi = b_0 + 0.5 * (yy - beta_vi' * inv_V_vi * beta_vi);

    % update q(alpha)
    d_vi = d_0 + (beta_vi'*beta_vi * a_vi / b_vi + trace(V_vi)) / 2;


    % calculate ELBO for convergence
    ELBO_old = ELBO_no_const;
    ELBO_no_const = - 0.5 * (a_vi / b_vi * (yy - 2 * beta_vi' * xy + beta_vi' * xx * beta_vi) + sum(sum(x .* (x * V_vi))))...
                    - my_logdet(inv_V_vi) / 2 - b_0 * a_vi / b_vi + gammaln(a_vi) - a_vi * log(b_vi) + a_vi ...
                    - gammaln(c_vi) - c_vi * log(d_vi);
    d_ELBO = ELBO_no_const - ELBO_old;
    to_save.ELBO = [to_save.ELBO ELBO_no_const];

    % iterations
    jcount = jcount + 1;
    fprintf('Iteration %d: ELBO = %.3f\n', jcount, ELBO_no_const);

end

khat = beta_vi; % final estimate (posterior mean after last update)
hprs.nsevar = b_vi / a_vi;
hprs.alpha = c_vi / d_vi / hprs.nsevar;



if jcount < opts.maxiter
    fprintf('Finished autoRidgeRegression in #%d steps\n', jcount);
else
    fprintf(1, 'Stopped autoRidgeRegression: MAXITER (%d) steps; dparams last step: %f\n', ...
        jcount, dparams);
end

% Put hyperparameter estimates into struct
hprs.ELBO = to_save.ELBO;