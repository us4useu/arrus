% Reconstructs rf image from raw rf and for rx aperture covering all the probe elements
function[recPre] = reconstructRfImgPart1(sys,acq,proc)
% Image reconstruction: delay & sum algorithm.
% 
% Outputs:
% 
% recPre            - precalculated parameters needed for reconstruction:
% recPre.iqEnable	- [logical] is the rf signal iq demodulated?
% recPre.iSamp      - [sample] (zSize,xSize,nRx,nTx) samples to pick
% recPre.modSig     - [] (zSize,xSize,nRx,nTx) re-modulation signal
% recPre.wghRx      - [] (zSize,xSize,nRx) transmit weights
% recPre.wghTx      - [] (zSize,xSize,nTx) receive weights
% 
% Inputs:
% 
% sys                       - system-related parameters
% sys.nElem                 - [elem] number of probe elements
% sys.pitch                 - [m] transducer pitch
% 
% acq.type                  - 'sta' or 'pwi' for STA scan or PWI scan
% acq.rxSampFreq            - [Hz] sampling frequency
% acq.txFreq                - [Hz] carrier (nominal) frequency
% acq.txNPer                - [] number of periods in the emitted pulse
% acq.c                     - [m/s] speed of sound used in calculation of tx delays
% acq.txApCent              - [m] x-positions of tx aperture center
% acq.txFoc                 - [m] focal length for STA scheme
% acq.txAng                 - [rad] tilting angles for PWI scheme
% 
% proc.dec                  - [] decimation factor
% proc.iqEnable             - [logical] 
% proc.xGrid                - [m] (1,xSize) x-grid vector for output rf image
% proc.zGrid                - [m] (1,zSize) z-grid vector for output rf image
% proc.txApod               - [] number of sigmas in the gaussian window used in tx apodization (0 -> rect. window)
% proc.rxApod               - [] number of sigmas in the gaussian window used in rx apodization (0 -> rect. window)

fs      = acq.rxSampFreq/proc.dec;
nSamp	= acq.nSamp/proc.dec;

xElem	= reshape(sys.xElem,1,1,[]);
iRx     = reshape(1:acq.rxApSize,1,1,[]);
iTx     = reshape(1:acq.nTx,1,1,1,[]);
xElemRx = (reshape(acq.rxApOrig,1,1,1,[]) - 1 + iRx - (sys.nElem-1)/2 - 1) * sys.pitch;

maxTang	= tan(asin(min(1,(acq.c/acq.txFreq*2/3)/sys.pitch)));  % 2/3*Lambda/pitch -> -6dB

%% initial delays
fstSampDel	= acq.startSample/acq.rxSampFreq;	% [s] rx delay with respect to start of tx
burstFactor	= acq.txNPer/(2*acq.txFreq);     % [s] burst factor
initDel     = - fstSampDel + acq.txDelCent + burstFactor;	% [s] total init delay

%% GPU memory allocation
if proc.gpuEnable
    iRx         = gpuArray(iRx);
    iTx         = gpuArray(iTx);
    xElem       = gpuArray(xElem);
    xElemRx     = gpuArray(xElemRx);
    proc.zGrid	= gpuArray(proc.zGrid);
    proc.xGrid	= gpuArray(proc.xGrid);
end

%% Precalculate tx delays and apodization
switch acq.type
    case 'sta'
        % synthetic transmit aperture method
        txFoc	= reshape(acq.txFoc,1,1,1,[]);
        xFoc	= reshape(acq.txFoc.*sin(acq.txAng) + acq.txApCent,1,1,1,[]);
        zFoc	= reshape(acq.txFoc.*cos(acq.txAng),1,1,1,[]);
        
        txDist	= sqrt((proc.zGrid' - zFoc).^2 + (proc.xGrid - xFoc).^2);
        txDist	= txDist.*sign(proc.zGrid' - zFoc) + txFoc;          % WARNING: sign()=0 => invalid txDist value
        
        txTang	= abs((proc.xGrid - xFoc)) ./ max(abs(proc.zGrid' - zFoc),1e-12);
        txApod	= double(txTang < maxTang);
%         txApod	= double(txTang < maxTang).*exp(-(txTang.^2)/(2*min(1e12,maxTang/proc.txApod)^2));
        
    case 'pwi'
        % plane wave imaging method
        txAng	= reshape(acq.txAng,1,1,1,[]);
        
        txDist	= (proc.xGrid - 0).*sin(txAng) + proc.zGrid'.*cos(txAng);
        
        % xElem: put the actual txAperture edges here
        r1      = (proc.xGrid-xElem(   1)).*cos(txAng) - proc.zGrid'.*sin(txAng);
        r2      = (proc.xGrid-xElem( end)).*cos(txAng) - proc.zGrid'.*sin(txAng);
        txApod	= double(r1 >= 0 & r2 <= 0);
        
end

%% Delay & Sum
rxDist	= sqrt((proc.xGrid-xElemRx).^2 + proc.zGrid'.^2);
if isfield(proc,'rxAngLim')
    proc.rxAngLim	= reshape(proc.rxAngLim,2,1,1,[]);
    rxAng	= atan((proc.xGrid-xElemRx)./ proc.zGrid');
    rxApod	= double(rxAng >= proc.rxAngLim(1,1,1,:) & rxAng <= proc.rxAngLim(2,1,1,:));
else
    rxTang	=  abs((proc.xGrid-xElemRx)   ./ proc.zGrid');
    rxApod	= double(rxTang < maxTang);
end

% calculate total delays
delTot	= (txDist + rxDist)/acq.c + initDel;	% [s]

% calculate sample numbers to be used in reconstruction (out-of-range sample numbers -> nSamp+1 -> sample=0)
iSamp	= delTot*fs + 1;                    % [samp]
iSamp(iSamp<1 | iSamp>nSamp)	= inf;
iSamp	= iSamp + ((iRx-1) + (iTx-1)*acq.rxApSize)*nSamp;	% [samp]

% calculate the rf samples (interpolated) and apodization weights
wghRx	= txApod.*rxApod;

% modulate if iq signal is used
if proc.iqEnable
    modSig	= exp(1i*2*pi*acq.txFreq*delTot);
end

% calculate weights for single tx
wghTx	= reshape(sum(wghRx,3),proc.zSize,proc.xSize,acq.nTx);

%% Output
recPre.iqEnable = proc.iqEnable;
recPre.iSamp	= iSamp;
recPre.modSig	= modSig;
recPre.wghRx	= wghRx;
recPre.wghTx	= wghTx;

end

