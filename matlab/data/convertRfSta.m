% Converts RF signal from SSTA format to other scan formats
function[rfOut] = convertRfSta(rfSta,mode,txFoc,txAp,txAng,sRej)

%% parameters
[nSamp,nElem,~]	= size(rfSta);

c0      = 1540e3;                 	%[mm/s]
pitch   = 0.3048;                   %[mm]
fs      = 40e6;                     %[Hz]

%% rejection of direct TX-RX signal
msk     = cumsum(double(rfSta ~= 0)) > 0;
msk     = [false(sRej,nElem,nElem); msk(1:end-sRej,:,:)];

rfSta	= rfSta.*msk;

%% tx delays
switch mode
    case 'pwi'
        nTx     = length(txAng);
        
        ap      = true(nElem,nTx);
        del     = (1:nElem).'*pitch.*sind(txAng)/c0;	%[s] [nElem nTx]
    case 'lin'
        % for even txAp, the aperture is half txAp +1 at start of the scan (+0 at its end)
        % txFoc is measured in z-direction no matter the txAng
        nTx     = nElem;
        
        ap      = logical(conv2(eye(nElem,nTx),[0;ones(txAp,1)],'same'));                 %[logical] [nElem nTx=nElem]
        del     = -sqrt(txFoc^2 + (((1:nElem).' - (1:nTx))*pitch - tand(txAng)*txFoc ).^2)/c0;	%[s] [nElem nTx=nElem]
end

del     = del - min(del.*ap);           %[s] [nElem nTx=nElem] delays with respect to the first active element of tx aperture
ds      = del*fs;                       %[samp] [nTx nLine nAng][nTx nAng]

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















