function [gamma2_proj, gamma2_proj_nz, gamma1_co_supp_hat] = HolisticPursuit(y, MLSC_i, ADMM_str, saveTrack)
% This function estimate gamma2 from the noisy vector y
% This algorithm alternates between the synthesis and the analysis
% constraints of the model.
% Every iteration contains an estimation of gamma2 using the effective
% model: min 1/2 ||y-D1*D2*gamma2||_2^2 + lambda1 ||gamma2||_1 
% but with a constraint that Phi*gamma2 = 0, where Phi consists of gamma1's
% cosupport which have already been estimated.
% Then we use gamma2 estimation to find another cosupport element in gamma1
% and we build Phi correspondingly.

%% 0. Init

% saveTrack = 1 <--> save the results for each co support options up to l1

if ~exist('saveTrack') || isempty(saveTrack)
    saveTrack = 0;
end

gamma1_co_supp_hat = zeros(MLSC_i.l1,1);

K = eye(MLSC_i.m2); % K spans the kernel subspace of Phi

if saveTrack == 0
    ADMM_str.precision = 2;
else
    ADMM_str.precision = 1;
    gamma2_proj_track = zeros(MLSC_i.m2, MLSC_i.l1 + 1);
end

%% 1. Main Loop
for i = 1:MLSC_i.l1
    %%% 1.1. Synthesis Part
    % Use the effective model
    % Estimate gamma2 with constraint of Phi*gamma2 = 0 using the matrix K
    [~, gamma2_proj, ADMM_str] = admm_lasso_constraint(y, MLSC_i.D_eff, K, ADMM_str);
    
    if saveTrack
        gamma2_proj_track(:,i) = gamma2_proj;
    end
    
    %%% 1.2. Analysis Part
    % Use the analysis constraints
    % Estimate new zero in gamma1 using the estimation of gamma2
    gamma1_propagated = MLSC_i.D2*gamma2_proj; % estimation of gamma1
    gamma1_supp_i = (1: MLSC_i.m1)';
    gamma1_supp_i( gamma1_co_supp_hat(1:(i-1)) ) = [];
    % Pick the minimal element in gamma1 and set it as a zero
    [~, lambda_i_tmp] = min( abs( gamma1_propagated( gamma1_supp_i ) ));
    lambda_i = gamma1_supp_i(lambda_i_tmp); % Set the location of the new zero
    
    % Build the K matrix with the new zero
    gamma1_co_supp_hat(i) = lambda_i;
    [~,~,v] = svd( MLSC_i.D2( gamma1_co_supp_hat(1:i), :) );
    K = v( :, (i+1):end);
    
end

%% 2. Final Estimation

ADMM_str.precision = 1; % Precise
% Estimate gamma2 with the estimated K
[~, gamma2_proj, ~] = admm_lasso_constraint(y, MLSC_i.D_eff, K, ADMM_str);

if saveTrack
    gamma2_proj_track(:,end) = gamma2_proj;
    gamma2_proj = gamma2_proj_track;
end
gamma2_proj_nz = Tools.nnz_thresh(gamma2_proj);

end
