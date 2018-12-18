function PrintErrorResults(varargin)
nMethods = nargin/3;
mse_i = (1:3:nargin);
contained = (2:3:nargin);
columns_name = {varargin{(3:3:nargin)}};
row_name_mse = {'gamma1 MSE'; 'gamma2 MSE'};
row_name_nz = {'gamma1 contained'; 'gamma2 contained'};
T_Mse = table(varargin{mse_i},'RowNames',row_name_mse,'VariableNames',columns_name);
disp(T_Mse);
T_nz = table(varargin{contained},'RowNames',row_name_nz,'VariableNames',columns_name);
disp(T_nz)
end