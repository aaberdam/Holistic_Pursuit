function [D, mean_error, mean_cardinality] = ...
    unitary_dictionary_learning(Y, D_init, num_iterations, pursuit_param)
% UNITARY_DICTIONARY_LEARNING Train a unitary dictionary via 
% Procrustes analysis.
%
% Inputs:
%   Y              - A matrix that contains the training patches 
%                    (as vectors) as its columns
%   D_init         - Initial UNITARY dictionary
%   num_iterations - Number of dictionary updates
%   pursuit_param  - The stopping criterion for the pursuit algorithm
%
% Outputs:
%   D          - The trained UNITARY dictionary
%   mean_error - A vector, containing the average representation error,
%                computed per iteration and averaged over the total 
%                training examples
%   mean_cardinality - A vector, containing the average number of nonzeros,
%                      computed per iteration and averaged over the total 
%                      training examples
   
 
% Allocate a vector that stores the average representation
% error per iteration
mean_error = zeros(num_iterations,1);
 
% Allocate a vector that stores the average cardinality per iteration
mean_cardinality = zeros(num_iterations,1);
 
% TODO: Set the dictionary to be D_init
% Write your code here... D = ???;
D = D_init;

 
% Run the Procrustes analysis algorithm for num_iterations
for iter = 1 : num_iterations
    
    % Compute the representation of each noisy patch
    [X, A] = batch_thresholding(D, Y, pursuit_param);
    
    % Compute and display the statistics
    fprintf('Iter %02d: ', iter);
    [mean_error(iter), mean_cardinality(iter)] = ...
        compute_stat(X, Y, A);
 
    % TODO: Update the dictionary via Procrustes analysis.
    % Solve D = argmin_D || Y - DA ||_F^2 s.t. D'D = I,
    % where 'A' is a matrix that contains all the estimated coefficients,
    % and 'Y' contains the training examples. Use the Procrustes algorithm.
    % Write your code here... D = ???;.
    [U,~,V] = svd(A * Y.');
    D = V * U.';
     
end
 
 
end
 
