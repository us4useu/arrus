% Reconstructs rf image from raw rf and for rx aperture covering all the probe elements
function[rfBfr,rfTx] = reconstructRfImg(rfRaw,sys,acq,proc)
% Image reconstruction: delay & sum algorithm.
% 
% Outputs:
% rfBfr                     - (zSize,xSize) output beamformed rf
% 
% Inputs:
% rfRaw                     - (nSamp,nRx,nTx) raw rf data
%                           rx aperture must cover all the probe elements;
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

% TODO
% apodization, STA step/focus/angle
    fstSampShift = -240 + acq.startSample;	% [samp] number of the sample that reflects the tx start

[nSamp,nRx,nTx] = size(rfRaw);
zSize	= length(proc.zGrid);
xSize	= length(proc.xGrid);

xElem	= (-(sys.nElem-1)/2:(sys.nElem-1)/2)*sys.pitch;
xElem	= reshape(xElem,1,1,[]);

fs      = acq.rxSampFreq/proc.dec;

maxTang	= tan(asin(min(1,(acq.c/acq.txFreq*2/3)/sys.pitch)));  % 2/3*Lambda/pitch -> -6dB

%% initial delays
fstSampDel	= fstSampShift/acq.rxSampFreq;	% [s] rx delay with respect to start of tx
burstFactor	= acq.txNPer/(2*acq.txFreq);     % [s] burst factor
initDel     = - fstSampDel + acq.txDelCent + burstFactor;	% [s] total init delay

%% GPU memory allocation
if isa(rfRaw,'gpuArray')
    xElem           = gpuArray(xElem);
    proc.zGrid	= gpuArray(proc.zGrid);
    proc.xGrid	= gpuArray(proc.xGrid);
    
    rfTx	= zeros(zSize,xSize,nTx,'gpuArray');
    wghTx	= zeros(zSize,xSize,nTx,'gpuArray');
else
    rfTx	= zeros(zSize,xSize,nTx);
    wghTx	= zeros(zSize,xSize,nTx);
end

%% Precalculate tx delays and apodization
switch acq.type
    case 'sta'
        % synthetic transmit aperture method
        txFoc	= reshape(acq.txFoc,1,1,[]);
        xFoc	= reshape(acq.txFoc.*sin(acq.txAng) + acq.txApCent,1,1,[]);
        zFoc	= reshape(acq.txFoc.*cos(acq.txAng),1,1,[]);
        
        txDist	= sqrt((proc.zGrid' - zFoc).^2 + (proc.xGrid - xFoc).^2);
        txDist	= txDist.*sign(proc.zGrid' - zFoc) + txFoc;          % WARNING: sign()=0 => invalid txDist value
        
        txTang	= abs((proc.xGrid - xFoc)) ./ max(abs(proc.zGrid' - zFoc),1e-12);
        txApod	= double(txTang < maxTang);
%         txApod	= double(txTang < maxTang).*exp(-(txTang.^2)/(2*min(1e12,maxTang/proc.txApod)^2));
        
    case 'pwi'
        % plane wave imaging method
        txAng	= reshape(acq.txAng,1,1,[]);
        
        txDist	= (proc.xGrid - 0).*sin(txAng) + proc.zGrid'.*cos(txAng);
        
        r1      = (proc.xGrid-xElem(   1)).*cos(txAng) - proc.zGrid'.*sin(txAng);
        r2      = (proc.xGrid-xElem( end)).*cos(txAng) - proc.zGrid'.*sin(txAng);
        txApod	= double(r1 >= 0 & r2 <= 0);
%         txTang	= tan(txAng);
%         txApod	= double(r1 >= 0 & r2 <= 0).*exp(-(txTang.^2)/(2*min(1e12,maxTang/proc.txApod)^2));
        
end

%% Precalculate rx delays and apodization
rxDist	= sqrt((proc.xGrid-xElem).^2 + proc.zGrid'.^2);
rxTang	=  abs((proc.xGrid-xElem)   ./ proc.zGrid');
rxApod	= double(rxTang < maxTang);
% rxApod	= double(rxTang < maxTang).*exp(-(rxTang.^2)/(2*min(1e12,maxTang/proc.rxApod)^2));

iRx     = 1:nRx;
if isa(rfRaw,'gpuArray')
    iRx	= gpuArray(iRx);
end
iRx     = reshape(iRx,1,1,[]);

%% Delay & Sum
for iTx=1:nTx
    % calculate total delays
    delTot	= (txDist(:,:,iTx) + rxDist)/acq.c + initDel;	% [s]
    
    % calculate sample numbers to be used in reconstruction (out-of-range sample numbers -> nSamp+1 -> sample=0)
    iSamp	= delTot*fs + 1;                    % [samp]
    iSamp(iSamp<1 | iSamp>nSamp)	= inf;
    iSamp	= iSamp + (iRx-1)*nSamp;            % [samp]
    
    % calculate the rf samples (interpolated) and apodization weights
    rfRx	= interp1(reshape(rfRaw(:,:,iTx),[],1),iSamp,'linear',0);
    wghRx	= txApod(:,:,iTx).*rxApod;
    
    % modulate if iq signal is used
    if proc.iqEnable
        rfRx	= rfRx.*exp(1i*2*pi*acq.txFreq*delTot);
    end
    
    % calculate rf and weights for single tx
    rfTx(:,:,iTx)	= sum(rfRx.*wghRx,3);
    wghTx(:,:,iTx)	= sum(wghRx,3);
end

% calculate the final rf image
rfBfr	= sum(rfTx,3)./sum(wghTx,3);

end

