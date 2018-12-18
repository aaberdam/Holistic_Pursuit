function [gamma1_hat_opt,FitInfo1] = Lasso_with_check( D1, y, s1)
[gamma1_hat_opt,FitInfo1] = lasso( D1, y);
lmbda_upper = FitInfo1.Lambda(1);
while Tools.nnz_thresh( gamma1_hat_opt(:,1) ) < s1
    new_lambda = lmbda_upper * 10 .^ logspace(  -5, 0, 50);
    lmbda_upper = new_lambda(1);
    [gamma1_hat_opt,FitInfo1] = lasso( D1, y, 'Lambda', new_lambda);
end

end