function unbiased_err = Unbiased_err(y, D_eff, gamma2)
[~, gamma2_nz_ind] = Tools.nnz_thresh(gamma2);
d_eff_s = D_eff(:, gamma2_nz_ind );
unbiased_err = mean( (y - d_eff_s * (d_eff_s  \ y) ).^2 );
end