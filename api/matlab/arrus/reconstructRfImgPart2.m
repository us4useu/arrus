% Reconstructs rf image from raw rf and for rx aperture covering all the probe elements
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
% recPre.iqEnable	- [logical] is the rf signal iq demodulated?
% recPre.iSamp      - [sample] (zSize,xSize,nRx,nTx) samples to pick
% recPre.modSig     - [] (zSize,xSize,nRx,nTx) re-modulation signal
% recPre.wghRx      - [] (zSize,xSize,nRx) transmit weights
% recPre.wghTx      - [] (zSize,xSize,nTx) receive weights

%% Delay & Sum
% calculate the rf samples (interpolated)
rfRx	= interp1(reshape(rfRaw,[],1),recPre.iSamp,'linear',0);

% calculate rf for single tx
rfRx	= rfRx.*recPre.modNWghRx;
rfTx	= reshape(sum(rfRx,3),size(recPre.wghTx));

% calculate the final rf image
rfBfr	= sum(rfTx,3)./sum(recPre.wghTx,3);

end

