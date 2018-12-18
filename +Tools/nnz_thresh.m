function [n, ind] = nnz_thresh(x)
thresh = 1e-3;
ind =  ( abs(x)./max(abs(x)) >= thresh);
n = sum( ind, 1);

end