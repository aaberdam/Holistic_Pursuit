function [mu] = MutualCoherence(A)
A_norm = normc(A);
gram_mat = A_norm.' * A_norm;
mu = max(max(gram_mat - eye(size(gram_mat,1))));
end