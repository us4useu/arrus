% Converts RF signal from SSTA format to other scan formats
function[rfOut] = convertRfSta(rfSta,sys,acq)
% Outputs:
% 
% rfOut                     - (nSamp,nElem,nTx) output raw rf data
% 
% 
% Inputs:
% 
% rfSta                     - (nSamp,nElem,nElem) input raw rf SSTA data
% 
% sys                       - system-related parameters
% sys.pitch                 - [m] transducer's pitch
% 
% acq                       - acquisition-related parameters
% acq.mode                  - 'pwi' or 'lin' for plane wave imaging or linear scan respectively
% acq.speedOfSound          - [m/s] speed of sound
% acq.tx.focus              - [m] tx focal depth (used in 'lin' mode only)
% acq.tx.apertureSize       - [elem] tx aperture (used in 'lin' mode only)
% acq.rx.apertureSize       - [elem] rx aperture (used in 'lin' mode only)
% acq.tx.angle              - [rad] (1,nTx or 1) tx angles (vector for 'pwi' mode, otherwise scalar)
% acq.rx.samplingFrequency	- [Hz] sampling frequency

%% parameters
[nSamp,nElem,~]	= size(rfSta);

nSampReject = 0; % [samp] number of input rf samples to be rejected (limits the shallow artifacts)

%% rejection of direct TX-RX signal
msk     = cumsum(double(rfSta ~= 0)) > 0;
msk     = [false(nSampReject,nElem,nElem); msk(1:end-nSampReject,:,:)];

rfSta	= rfSta.*msk;

%% tx delays
switch acq.mode
    case 'pwi'
        nTx     = length(acq.tx.angle);
        
        ap      = true(nElem,nTx);
        del     = (1:nElem).'*sys.pitch.*sin(acq.tx.angle)/acq.speedOfSound;	% [s] [nElem nTx]
    case 'lin'
        % for even acq.tx.apertureSize, the aperture is half apertureSize+1 at start of the scan (+0 at its end)
        % acq.tx.focus is measured in z-direction no matter the txAng
        nTx     = nElem;
        
        ap      = logical(conv2(eye(nElem,nTx),[0;ones(acq.tx.apertureSize,1)],'same'));	% [logical] [nElem nTx=nElem]
        del     = -sqrt(acq.tx.focus^2 + (((1:nElem).' - (1:nTx))*sys.pitch - tan(acq.tx.angle)*acq.tx.focus ).^2) ...
                / acq.speedOfSound;	% [s] [nElem nTx=nElem]
end

del     = del - min(del.*ap);	% [s] [nElem nTx=nElem] delays with respect to the first active element of tx aperture
ds      = del*acq.rx.samplingFrequency;            % [samp] [nTx nLine nAng][nTx nAng]

%% rf synthesis (with linear interpolation)
rfOut	= zeros(nSamp,nElem,nTx);
wb      = waitbar(0,'convertRfSta');
for iTx=1:nTx
    for iElem=1:nElem
        if ~ap(iElem,iTx)
            continue;
        end
        
        sOut	= (1+ceil(ds(iElem,iTx))) : nSamp;
        sIn     = sOut - ds(iElem,iTx);
        
        rfOut(sOut,:,iTx)	= rfOut(      sOut,:,iTx) + ...
                              rfSta(floor(sIn),:,iElem)*   mod(ds(iElem,iTx),1) + ...
                              rfSta( ceil(sIn),:,iElem)*(1-mod(ds(iElem,iTx),1));
    end
    waitbar(((iTx-1)*nElem+iElem)/(nTx*nElem),wb);
end
close(wb);

%% limit rx aperture for LIN mode (limited, scanning aperture)
if strcmp(acq.mode,'lin')
    rfOutAux = rfOut;
    
    rxApertCent	= 1:nTx;
    rxApertFst	= rxApertCent - floor((acq.rx.apertureSize-1)/2);
    rxApertLst	= rxApertCent +  ceil((acq.rx.apertureSize-1)/2);
    
    rfOut	= zeros(nSamp,acq.rx.apertureSize,nTx);
    for iTx=1:nTx
        rxApertFullPart	=    max(1,  rxApertFst(iTx))  :          min(nElem,rxApertLst(iTx));
        rxApertScanPart	= (1+max(0,1-rxApertFst(iTx))) : (acq.rx.apertureSize-max(0,rxApertLst(iTx)-nElem));
        rfOut(:,rxApertScanPart,iTx) = rfOutAux(:,rxApertFullPart,iTx);
    end
    
end

end













