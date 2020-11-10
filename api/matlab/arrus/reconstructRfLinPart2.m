% Reconstructs rf image lines from raw rf data using precalculated parameters
function[rfBfr] = reconstructRfLinPart2(rfRaw,recPre)
% Image reconstruction: delay & sum algorithm.
% 
% Outputs:
% rfBfr         - (nSamp,nTx)  output beamformed rf
% 
% Inputs:
% rfRaw         - (nSamp,nRx,nTx) raw rf data;
%                   tx & rx apertures must be centered at the intersection of imaging line & probe surface;
%                   tx time delay of the tx aperture center element (txCentDel) must be constant for all tx's;
%                   if rfRaw is gpuArray then calculations are done on GPU;
% 
% recPre            - precalculated parameters needed for reconstruction:
% recPre.iSamp0     - [sample] (zSize,xSize,nRx,nTx) samples to pick
% recPre.iSamp      - [sample] (zSize,xSize,nRx,nTx) samples to pick
% recPre.modNWghRx	- [] (zSize,xSize,nRx,nTx) re-modulation signal * receive weights

%% Delay & Sum
[nSamp,nRx,~] = size(recPre.modNWghRx);
nTx     = numel(rfRaw)/nSamp/nRx;

% calculate the rf samples (interpolated)
rfRaw	= reshape(rfRaw,length(recPre.iSamp0),[]);
rfBfr	= interp1(recPre.iSamp0(:),rfRaw,recPre.iSamp(:),'linear',0);	% WARNING -> see the comment at the end of the script
rfBfr	= reshape(rfBfr,[nSamp,nRx,nTx]);

% calculate rf for each single line
rfBfr	= rfBfr.*recPre.modNWghRx;
rfBfr	= reshape(sum(rfBfr,2),[nSamp,nTx]);

% WARNING
% The first argument in interp1 function is optional, code could be: rfBfr	= interp1(rfRaw,iSamp,'linear',0);
% However, if interp1 is executed on GPU (rfRaw is gpuArray), then interp1 contains a bug which results in CUDA error.
% The solution is to keep the first argument of the interp1 function.
% Solution found here: https://uk.mathworks.com/matlabcentral/answers/462545-interp1-gpuarray-bug

end
























