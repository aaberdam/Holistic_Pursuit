function [gamma_se_err, gamma_sup_err] = ErrorEvaluation(varargin)
% gamma1_hat, gamma2_hat, gamma1, gamma2
% Or:
% gamma2_hat, gamma2
if nargin == 4
    gamma1_hat = varargin{1};
    gamma2_hat = varargin{2};
    gamma1 = varargin{3};
    gamma2 = varargin{4};
    diff_gamma1 = gamma1 - gamma1_hat;
    diff_gamma2 = gamma2 - gamma2_hat;
    gamma_se_err = [sum((diff_gamma1).^2); sum((diff_gamma2).^2)];
    
    % gamma_nz_err_hat = [Tools.nnz_thresh(diff_gamma1); Tools.nnz_thresh(diff_gamma2)];
    
    [s1_hat, gamma1_hat_ind] = Tools.nnz_thresh(gamma1_hat);
    [s1, gamma1_ind] = Tools.nnz_thresh(gamma1);
    
    [s2_hat, gamma2_hat_ind] = Tools.nnz_thresh(gamma2_hat);
    [s2, gamma2_ind] = Tools.nnz_thresh(gamma2);
    
    % gamma_containd_supp_hat = [ ( sum(gamma1_hat_ind .* gamma1_ind) == s1_hat);
    %                             ( sum(gamma2_hat_ind .* gamma2_ind) == s2_hat)];
    gamma_sup_err = [ ( sum(gamma1_hat_ind .* gamma1_ind) / max(s1_hat, s1));
        ( sum(gamma2_hat_ind .* gamma2_ind) / max(s2_hat, s2))];
elseif nargin==2
    gamma2_hat = varargin{1};
    gamma2 = varargin{2};
    diff_gamma2 = gamma2 - gamma2_hat;
    gamma_se_err = sum((diff_gamma2).^2) ./ sum(gamma2.^2);
    [s2_hat, gamma2_hat_ind] = Tools.nnz_thresh(gamma2_hat);
    [s2, gamma2_ind] = Tools.nnz_thresh(gamma2);
    gamma_sup_err = ( sum(gamma2_hat_ind .* gamma2_ind, 1) ./ max(s2_hat, s2));
end
end