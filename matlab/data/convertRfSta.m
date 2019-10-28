% Converts RF signal from SSTA format to other scan formats
function[rfOut] = convertRfSta(rfSta,pitch,fSamp,sos,mode,txFoc,txAp,txAng,sRej)
% rfOut     - [] (nSamp,nElem,nTx) output raw rf data

% rfSta     - [] (nSamp,nElem,nElem) input raw rf SSTA data
% pitch     - [m] transducer's pitch
% fSamp     - [Hz] sampling frequency
% sos       - [m/s] speed of sound
% mode      - 'pwi' or 'lin' for plane wave imaging or linear scan respectively
% txFoc     - [m] tx focal depth (used in 'lin' mode only)
% txAp      - [elem] tx aperture (used in 'lin' mode only)
% txAng     - [deg] (1,nTx or 1) tx angles (vector for 'pwi' mode, otherwise scalar)
% sRej      - [samp] number of input rf samples to be rejected (limits the shallow artifacts)

%% parameters
[nSamp,nElem,~]	= size(rfSta);

%% rejection of direct TX-RX signal
msk     = cumsum(double(rfSta ~= 0)) > 0;
msk     = [false(sRej,nElem,nElem); msk(1:end-sRej,:,:)];

rfSta	= rfSta.*msk;

%% tx delays
switch mode
    case 'pwi'
        nTx     = length(txAng);
        
        ap      = true(nElem,nTx);
        del     = (1:nElem).'*pitch.*sind(txAng)/sos;	% [s] [nElem nTx]
    case 'lin'
        % for even txAp, the aperture is half txAp +1 at start of the scan (+0 at its end)
        % txFoc is measured in z-direction no matter the txAng
        nTx     = nElem;
        
        ap      = logical(conv2(eye(nElem,nTx),[0;ones(txAp,1)],'same'));	% [logical] [nElem nTx=nElem]
        del     = -sqrt(txFoc^2 + (((1:nElem).' - (1:nTx))*pitch - tand(txAng)*txFoc ).^2)/sos;	% [s] [nElem nTx=nElem]
end

del     = del - min(del.*ap);	% [s] [nElem nTx=nElem] delays with respect to the first active element of tx aperture
ds      = del*fSamp;            % [samp] [nTx nLine nAng][nTx nAng]

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

end















