function [x, gamma, ADMM_str] = admm_lasso_constraint(y, D, K, ADMM_str)
%%
% We aim to solve the lasso constraint problem
% $\frac{1}{2} \left\Vert y - DKx \right\Vert_2^2 + \lambda \left\Vert \gamma
% \right\Vert_1 s.t. Kx = \gamma$
%
% Using the ADMM method we can write
% $\frac{1}{2} \left\Vert y - DKx \right\Vert_2^2 + \lambda \left\Vert \gamma
% \right\Vert_1 + \rho \frac{1}{2} \left\Vert Kx - \gamma + u \right\Vert_2^2$
%
% Which can be seperated to 3 steps:
%%
% * $\min_{\gamma} \rho \frac{1}{2} \left\Vert Kx + u - \gamma \right\Vert_2^2 + \lambda \left\Vert \gamma
% \right\Vert_1 \Rightarrow \gamma^{\star} = \mathcal{S}_{\lambda/\rho}(Kx + u)$
% * $\min_{x} \frac{1}{2} \left\Vert y - DKx \right\Vert_2^2 + \rho \frac{1}{2}
% \left\Vert Kx - \gamma + u \right\Vert_2^2 \Rightarrow
% x^{\star} = \left( (DK)^T DK + \rho I \right)^{\dagger} \left( (DK)^T y + \rho K^T (\gamma-u) \right)$
% * $u \gets u + \rho (Kx-\gamma)$
%
%%
% If admm_mode = 1 then Lambda doesn't change
% If admm_mode = 0 then Lambda may change such that cardinality of gamma
% equals thresh (=s2) (Default)
% If admm_mode = 2 the Lambda is set such that $\left\Vert y- D \gamma
% \right\Vert_2^2 \leq thresh $
%%
% If precision = 1 then the solution is very accuracy (1e5 iterations)
% (Default)
% If precision = 2 then the solution is very accuracy (5e3 iterations)
%% Initialize
% lambda_old
if isempty(ADMM_str.precision)
    ADMM_str.precision = 1;
end
if isempty(ADMM_str.admm_mode)
    ADMM_str.admm_mode = 0;
end

if ADMM_str.precision == 1
    maxItr = 1e5;
else
    maxItr = 5e3;
end

% ToDo change the init to zeros
x = randn(size(K,2),1);
gamma = randn(size(D,2),1);
u = randn(size(D,2),1);

lambda_new = ADMM_str.lambda;
lambda_old = lambda_new;

%% Lambda may change
if (ADMM_str.admm_mode == 0) || (ADMM_str.admm_mode == 2)
    if ADMM_str.admm_mode == 0
        % Lambda is set such that nnz(gamma) = s
        s = ADMM_str.thresh;
        nnz_t = @(gamma) Tools.nnz_thresh(gamma);
        continue_func = @(gamma) abs( nnz_t(gamma) - s ) > 2;
        too_dense = @(gamma) nnz_t(gamma) > s;
        too_sparse = @(gamma) nnz_t(gamma) < s;
    else
        % Lambda is set such that sum( (y - D * gamma).^2) = noise_level
        noise_level = ADMM_str.thresh;
        % 5% far from threshold
        continue_func = @(gamma) abs( sum( (y - D * gamma).^2) / noise_level - 1) > 0.05;
        too_dense = @(gamma) sum( (y - D * gamma).^2) < noise_level;
        too_sparse = @(gamma) sum( (y - D * gamma).^2) > noise_level;
    end
    
    lower = 0;
    upper = 2 * lambda_new;
    upper_tested = 0;
    
    max_lambda_itr = 20;
    lambda_itr = 0;
    pinv_mat = pinv((D*K)' * (D*K) + ADMM_str.rho * eye(size(x,1)));
    
    while continue_func(gamma) 
        lambda_old = lambda_new;
        lambda_itr = lambda_itr +1;
        
        soft_thresh = @(x) ( abs(x) > (lambda_old/ADMM_str.rho) ) .* ( abs(x) - (lambda_old/ADMM_str.rho) ) .* sign(x);
        DKY = (D*K)' * y;
        
        gamma = soft_thresh( K*x + u);
        
        %%% Iterations
        [ K, x, gamma, u, itr] = admm_iterations(soft_thresh, pinv_mat, DKY, ADMM_str.rho, K, x, gamma, u, y, maxItr);
        
%         if admm_mode == 0
%             fprintf('Iteration %d:    nnz(gamma) = %d, inner_itr = %d \n',lambda_itr,nnz_t(gamma), itr);
%         elseif admm_mode == 2
%             fprintf('Iteration %d:    MSE/thresh = %.2f, inner_itr = %d \n',lambda_itr,sum( (y - D * gamma).^2) / noise_level, itr);
%         end
        
        if too_sparse(gamma)
            upper = lambda_old;
            upper_tested = 1;
            lambda_new = (lambda_old + lower) / 2;
        end
        if too_dense(gamma)
            lower = lambda_old;
            if upper_tested == 1
                lambda_new = (lambda_old + upper) / 2;
            else
                lambda_new = upper;
                upper_tested = 0;
                upper = 2 * lambda_new;
            end
        end
        
        if lambda_itr == max_lambda_itr
            warning('The lambda of the ADMM did not converge with %d iterations \n', max_lambda_itr)
            break;
        end
    end
    lambda_new = lambda_old;
    
    
    
    %% Lambda doesn't change
elseif ADMM_str.admm_mode == 1
    
    pinv_mat = pinv((D*K)' * (D*K) + ADMM_str.rho * eye(size(x,1)));
    soft_thresh = @(x) ( abs(x) > (lambda_old/ADMM_str.rho) ) .* ( abs(x) - (lambda_old/ADMM_str.rho) ) .* sign(x);
    DKY = (D*K)' * y;
    
    %%% Iterations
    [ K, x, gamma, u, itr] = admm_iterations(soft_thresh, pinv_mat, DKY, ADMM_str.rho, K, x, gamma, u, y, maxItr);
    
    
    % fprintf('ADMM:    nnz(gamma) = %d, inner_itr = %d \n',nnz_t(gamma), itr);
    
end

ADMM_str.lambda = lambda_new;

end

function [ K, x, gamma, u, itr] = admm_iterations(soft_thresh, pinv_mat, DKY, rho, K, x, gamma, u, y, maxItr)
itr = 0;

while sum((K * x - gamma).^2) / sum(u.^2) > 1e-4
    itr = itr + 1;
    gamma = soft_thresh( K*x + u);
    x = pinv_mat * ( DKY + rho * K' * ( gamma - u ) );
    u = u + rho * ( K * x - gamma );
    if itr == maxItr
        warning('The ADMM did not converge with %d iterations \n', maxItr)
        break;
    end
end
end