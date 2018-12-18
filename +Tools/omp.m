function x = omp(A, b, k)
% OMP Solve the P0 problem via OMP
%
% Solves the following problem:
%   min_x ||b - Ax||_2^2 s.t. ||x||_0 <= k
%
% The solution is returned in the vector x

% Initialize the vector x
x = zeros(size(A,2),1);
A_norm_inv = 1 ./ sum(A.^2,1).';
% TODO: Implement the OMP algorithm
% Write you code here... x = ????;
res = zeros(numel(b), k+1);
res(:,1) = b;
supp = zeros(k,1);
for i = 1:k
    [~, supp(i)] = max( ( A.' * res(:, i) ).^2 .* A_norm_inv );
    A_s = A(:, supp(1:i));
    x_k = zeros(size(A,2),1);
    x_k(supp(1:i)) = A_s \ b;
    res(:, i+1) = b - A*x_k;
end
x = x_k;
end

