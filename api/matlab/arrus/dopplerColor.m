% Calculates Doppler color map from a set of reconstructed iq images
function[colorMap,powerMap] = dopplerColor(iqImgSet,pri)
% 
% Outputs:
% colorMap                  - (zSize,xSize) output doppler color map [Hz]
% powerMap                  - (zSize,xSize) output doppler power map
% 
% Inputs:
% iqImgSet                  - (zSize,xSize,nTx) reconstructed iq image set
% pri                       - [s] pulse repetition interval

%% Clutter (regression) HP filter
nTx = size(iqImgSet,3);
iTx = reshape(-(nTx-1)/2:(nTx-1)/2, 1, 1, nTx);

meanVal = mean(iqImgSet, 3);
gradVal = sum(iqImgSet.*sign(iTx), 3)*4/(nTx^2-1);

iqImgSet = iqImgSet - meanVal - gradVal.*iTx;

%% mean frequency estimator
colorMap = angle( sum( iqImgSet(:,:,2:end) .* ...
                 conj( iqImgSet(:,:,1:end-1)), 3) ) / (2*pi*pri);	% [Hz]
powerMap = mean(iqImgSet .* conj(iqImgSet), 3);

end


