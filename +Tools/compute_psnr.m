function [ psnr_val ] = compute_psnr(y_original, y_estimated)
% COMPUTE_PSNR Compute the PSNR between two images
%
% Inputs:
%  y_original  - The original image
%  y_estimated - The estimated image
%
% Output:
%  psnr_val - The Peak Signal to Noise Ratio (PSNR) score

y_original = y_original(:);
y_estimated = y_estimated(:);

dynamic_range = 255;

% Compute the Mean Squared Error (MSE)
mse_val = 1/numel(y_original) * norm(y_original-y_estimated).^2;

% Compute the PSNR
psnr_val = 10*log10(dynamic_range^2 / mse_val);

end

