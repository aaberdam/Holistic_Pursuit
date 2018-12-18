function signMat = RandSign(varargin)
s = cell2mat(varargin);
signMat = 2*((rand(s)>0.5)-0.5);

end