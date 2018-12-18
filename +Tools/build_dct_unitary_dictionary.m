function DCT = build_dct_unitary_dictionary(patch_size)
% BUILD_DCT_UNITARY_DICTIONARY Creates an overcomplete 2D-DCT dictionary.
%
% Inputs:
% patch_size  - Atom size [height, width] (must satisfy height == width)
%
% Outputs:
% DCT - Unitary DCT dictionary with normalized columns

%% Make sure that the patch is square

if (patch_size(1) ~= patch_size(2))
    error('This only works for square patches'); 
end

nAtoms = prod(patch_size);


%% Create DCT for one axis

Pn = ceil(sqrt(nAtoms));
DCT = zeros(patch_size(1) , Pn);

for k = 0 : 1 : Pn - 1
    
	V = cos( (0.5 + (0 : 1 : patch_size(1) - 1) ) * k * pi / Pn);
	if (k > 0), V = V - mean(V); end;
	DCT(: , k+1) = V / norm(V);
    
end

%% Create the DCT for both axes

DCT = kron(DCT , DCT);


