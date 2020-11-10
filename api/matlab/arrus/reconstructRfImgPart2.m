% Reconstructs rf image from raw rf data using precalculated parameters
function[rfBfr,rfTx] = reconstructRfImgPart2(rfRaw,recPre)
% Image reconstruction: delay & sum algorithm.
% 
% Outputs:
% rfBfr             - (zSize,xSize) output beamformed rf (HRI)
% rfTx              - (zSize,xSize,nTx) output beamformed rf (LRI)
% 
% Inputs:
% rfRaw             - (nSamp,nRx,nTx) raw rf data
% 
% recPre            - precalculated parameters needed for reconstruction:
% recPre.iSamp      - [sample] (zSize,xSize,nRx,nTx) samples to pick
% recPre.modNWghRx	- [] (zSize,xSize,nRx,nTx) re-modulation signal * receive weights
% recPre.wghTx      - [] (zSize,xSize,nTx) transmit weights

%% Delay & Sum
% calculate the rf samples (interpolated)
rfRx	= interp1(reshape(rfRaw,[],1),recPre.iSamp,'linear',0);

% calculate rf for single tx
rfRx	= rfRx.*recPre.modNWghRx;
rfTx	= reshape(sum(rfRx,3),size(recPre.wghTx));

% calculate the final rf image
rfBfr	= sum(rfTx,3)./sum(recPre.wghTx,3);

end

