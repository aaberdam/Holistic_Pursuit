function [MLSC, x, gamma1, gamma2, Lambda_1_c, Lambda_1, Lambda_2] = CreateMlsrSignalD2Kernel(MLSC)
% MLSC.dict_type:
% 0 - Gaussian dictionary normalized columns,
% 1 - Gaussian dictionary normalized columns of D1, normalized rows of D2
% 2 - Diff matrix
% 3 - D1 Gaussian dictionary and D2 is a sparse matrix

% n,m1,m2,s2,s1,MLSC.sig_norm, MLSC.dict_type

%% Create the Dictionaries
if isempty(MLSC.D1)
    if isempty(MLSC.dict_type)
        MLSC.dict_type = 0;
    end
    
    if MLSC.dict_type == 0 % Gaussian dictionary normalized columns
        % D1
        MLSC.D1 = 1/sqrt(MLSC.n) * randn(MLSC.n, MLSC.m1);
        % D2
        MLSC.D2 = 1/sqrt(MLSC.m1) * randn(MLSC.m1, MLSC.m2);
        
        
    elseif MLSC.dict_type == 1 % Gaussian dictionary normalized columns of D1, normalized rows of D2
        % D1
        MLSC.D1 = 1/sqrt(MLSC.n) * randn(MLSC.n, MLSC.m1);
        % D2
        MLSC.D2 = 1/sqrt(MLSC.m2) * (randn(MLSC.m1, MLSC.m2));
        
        
    elseif MLSC.dict_type == 2 %Dif Matrix
        % D1
        MLSC.D1 = 1/sqrt(MLSC.n) * randn(MLSC.n, MLSC.m1);
        % D2
        MLSC.D2 = zeros(MLSC.m1,MLSC.m2);
        for i = 1:MLSC.m2
            ind = mod(i+(0:1)-1,MLSC.m2)+1;
            % Vertical
            MLSC.D2(ind,i) = [1;-1];
            % Horizonal
            MLSC.D2(MLSC.m2 + i, ind) = [1,-1];
        end
        
    elseif MLSC.dict_type == 3 %Gaussian & Sparse
        % D1
        MLSC.D1 = 1/sqrt(MLSC.n) * randn(MLSC.n, MLSC.m1);
        % D2
        MLSC.D2 = zeros(MLSC.m1,MLSC.m2);
        nnz_D2 =  round(0.05*MLSC.m1);
        for i = 1:MLSC.m2
            ind = randperm( MLSC.m1, nnz_D2);
            MLSC.D2(ind,i) = 1/sqrt(nnz_D2) * randn(nnz_D2,1);
        end
    end
    MLSC.D_eff = MLSC.D1 * MLSC.D2;
end


%% Create gamma2 & gamma1
if ~isempty(MLSC.s2)
    
    s1 = MLSC.m1 - MLSC.l1;
    Lambda_1 = randperm(MLSC.m1,s1); % Choose randomly gamma1's support
    Lambda_1_c = (1:MLSC.m1); % Cosupport of gamma1
    Lambda_1_c(Lambda_1) = [];
    
    Lambda_2 = randperm(MLSC.m2,MLSC.s2); % Choose randomly gamma1's support
    
    % Build the matrix Phi which is a submatrix from the dictioanry D2 with
    % rows from the cosupport of gamma1 and columns from the support of
    % gamma2, such that Phi*gamma2 = 0, and therefore, gamma2 has to be in
    % the kernel space of Phi
    Phi = MLSC.D2(Lambda_1_c, Lambda_2); 
    [~,~,v] = svd(Phi); % SVD for Phi
    % K is the matrix spans the kernel space of Phi
    K = v(:,rank(Phi)+1:end); 
    
    % Build gamma2 in the kernel space of Phi by gamma2 = K * alpha2 
    gamma2 = zeros(MLSC.m2,1); % init gamma2
    alpha2 = randn( size(K,2) , 1); % randomly choose alpha2 
    gamma2_nnz_values = sqrt(1/(size(K,2))) * K * alpha2; % normalize the energy
    gamma2( Lambda_2) = gamma2_nnz_values;
    
    gamma1 = MLSC.D2 * gamma2; % set gamma1 = D2 * gamma2
    
    x = MLSC.D1 * gamma1; % set x = D1 * gamma1
    
    % Ampitude
    a = MLSC.sig_norm / norm(x);
    x = a * x; % set the amplitude of the signal 
    gamma1 = a * gamma1; % set the amplitude of gamma1
    gamma2 = a * gamma2; % set the amplitude of gamma2
end

end