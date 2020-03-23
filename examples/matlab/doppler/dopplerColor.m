% Color Doppler
function[color,power] = dopplerColor(iq,prf)
% Color Doppler processing.
% 
% Outputs:
% color         - (zSize,xSize) output Doppler color map
% power         - (zSize,xSize) output Doppler power map
% 
% Inputs:
% iq            - (zSize,xSize,nRep) input iq signal
% prf           - [Hz] pulse repetition frequency (optional; if defined, then color is in Hz, otherwise, in radians)

nRep = size(iq,3);

% wall clutter filter (reg filter for now)
tAux = permute((-(nRep-1)/2:(nRep-1)/2),[1 3 2]);
grad = sum(iq.*sign(tAux),3)*4/(nRep^2-1);
trend  = mean(iq,3) + grad.*tAux;

iq = iq - trend;

% power & color estimates
color = angle(sum(iq(:,:,2:end).*conj(iq(:,:,1:(end-1))),3));
power = sum(iq.*conj(iq),3);

% convert radians to Hz
if nargin>1
    color = color*prf/(2*pi);
end

end
