% Reconstructs rf image from raw rf and for rx aperture covering all the probe elements
function[rfBfr] = reconstructRfImg(rfRaw,sys,acq,proc)
% Image reconstruction: delay & sum algorithm.
% 
% Outputs:
% rfBfr                     - (zSize,xSize) output beamformed rf
% 
% Inputs:
% rfRaw                     - (nSamp,nRx,nTx) raw rf data
%                           rx aperture must cover all the probe elements;
% sys                       - system-related parameters
% sys.nElements             - [elem] number of probe elements
% sys.pitch                 - [m] transducer pitch
% 
% acq.mode                  - 'sta' or 'pwi' for STA scan or PWI scan
% acq.rx.samplingFrequency	- [Hz] sampling frequency
% acq.tx.frequency          - [Hz] carrier (nominal) frequency
% acq.tx.nPeriods           - [] number of periods in the emitted pulse
% acq.speedOfSound          - [m/s] speed of sound used in calculation of tx delays
% acq.tx.apertureSize       - [elem] number of probe's elements forming the tx aperture
% acq.tx.focus              - [m] focal length for STA scheme
% acq.tx.angle              - [rad] tilting angles for PWI scheme
% 
    % proc.speedOfSound         - [m/s] speed of sound used in reconstruction
% proc.ddc.decimation       - [] decimation factor
% proc.ddc.iqEnable         - [logical] 
% proc.das.xGrid            - [m] (1,xSize) x-grid vector for output rf image
% proc.das.zGrid            - [m] (1,zSize) z-grid vector for output rf image

% TODO
% focDel, firstSampShift, gpu
fstSampShift = 0;	% [samp] number of the sample that reflects the tx start

[~,nRx,nTx] = size(rfRaw);
zSize	= length(proc.das.zGrid);
xSize	= length(proc.das.xGrid);

xElem	= (-(sys.nElements-1)/2:(sys.nElements-1)/2)*sys.pitch;
xElem	= permute(xElem,[1 3 2]);

fs      = acq.rx.samplingFrequency/proc.ddc.decimation;

%% initial delays
fstSampDel	= fstSampShift/acq.rx.samplingFrequency;                                       % [s] rx delay with respect to start of tx
burstFactor	= acq.tx.nPeriods/(2*acq.tx.frequency);                                           % [s] burst factor
if strcmp(acq.mode,'sta') && (acq.tx.focus > 0)
    % MSTA with focusing -> need to compensate for delay of 
    focDel	= (sqrt(((acq.tx.apertureSize-1)/2*sys.pitch).^2 + acq.tx.focus.^2) - acq.tx.focus)/acq.speedOfSound;     % [s] focusing delay of center of tx aperture
else
    % SSTA, MSTA with defocusing, PWI -> no focusing
    focDel	= 0;
end
initDel     = - fstSampDel + focDel + burstFactor;                      % [s] total init delay

%% GPU memory allocation
if isa(rfRaw,'gpuArray')
    xElem           = gpuArray(xElem);
    proc.das.zGrid	= gpuArray(proc.das.zGrid);
    proc.das.xGrid	= gpuArray(proc.das.xGrid);
    
    rfTx	= zeros(zSize,xSize,nTx,'gpuArray');
    wghTx	= zeros(zSize,xSize,nTx,'gpuArray');
else
    rfTx	= zeros(zSize,xSize,nTx);
    wghTx	= zeros(zSize,xSize,nTx);
end

%% Precalculate tx delays and apodization
switch acq.mode
    case 'sta'
        % synthetic transmit aperture method
        txDist	= sqrt((proc.das.zGrid' - acq.tx.focus).^2 + (proc.das.xGrid-xElem).^2);
        txDist	= txDist.*sign(proc.das.zGrid' - acq.tx.focus) + acq.tx.focus;          % WARNING: sign()=0 => invalid txDist value
        
        txFNum	= abs((proc.das.xGrid-xElem)) ./ max(abs(proc.das.zGrid' - acq.tx.focus),1e-12);
        txApod	= double(txFNum < 0.5);
        
    case 'pwi'
        % plane wave imaging method
        txAng	= permute(acq.tx.angle,[1 3 2]);
        eFst	= 1 + (sys.nElements-1)*(txAng<0);
        
        txDist	= (proc.das.xGrid-xElem(eFst))*sin(txAng) + proc.das.zGrid'*cos(txAng);
        
        r1      = (proc.das.xGrid-xElem(   1))*cos(txAng) - proc.das.zGrid'*sin(txAng);
        r2      = (proc.das.xGrid-xElem( end))*cos(txAng) - proc.das.zGrid'*sin(txAng);
        txApod	= double(r1 >= 0 & r2 <= 0);
        
end

%% Precalculate rx delays and apodization
rxDist	= sqrt((proc.das.xGrid-xElem).^2 + proc.das.zGrid'.^2);
rxFNum	=  abs((proc.das.xGrid-xElem)   ./ proc.das.zGrid');
rxApod	= rxFNum < 0.5;

iRx     = repmat(permute(1:nRx,[1 3 2]),[zSize xSize 1]);

%% Delay & Sum
wb = waitbar(0,'rf image reconstruction');
for iTx=1:nTx
    
%                     % calculate tx delays and apodization
%                     switch acq.mode
%                         case 'sta'
%                             % synthetic transmit aperture method
%                             txDist	= sqrt((proc.das.zGrid' - acq.tx.focus).^2 + (proc.das.xGrid-xElem(iTx)).^2);
%                             txDist	= txDist.*sign(proc.das.zGrid' - acq.tx.focus) + acq.tx.focus;          % WARNING: sign()=0 => invalid txDist value
% 
%                             txFNum	= abs((proc.das.xGrid-xElem(iTx))) ./ max(abs(proc.das.zGrid' - acq.tx.focus),1e-12);
%                             txApod	= double(txFNum < 0.5);
% 
%                         case 'pwi'
%                             % plane wave imaging method
%                             if acq.tx.angle(iTx) >= 0
%                                 eFst	= 1;
%                             else
%                                 eFst	= sys.nElements;
%                             end
%                             txDist	= (proc.das.xGrid-xElem(eFst))*sin(acq.tx.angle(iTx)) + proc.das.zGrid'*cos(acq.tx.angle(iTx));
% 
%                             r1      = (proc.das.xGrid-xElem(   1))*cos(acq.tx.angle(iTx)) - proc.das.zGrid'*sin(acq.tx.angle(iTx));
%                             r2      = (proc.das.xGrid-xElem( end))*cos(acq.tx.angle(iTx)) - proc.das.zGrid'*sin(acq.tx.angle(iTx));
%                             txApod	= double(r1 >= 0 & r2 <= 0);
% 
%                     end
    
    % calculate total delays
    delTot	= (txDist(:,:,iTx) + rxDist)/acq.speedOfSound + initDel;	% [s]
    
    % calculate sample numbers to be used in reconstruction (out-of-range sample numbers -> nSamp+1 -> sample=0)
    iSamp	= delTot*fs + 1;                 % [samp]
    
    % calculate the rf samples (interpolated) and apodization weights
    rfRx	= interp2(rfRaw(:,:,iTx),iRx,iSamp,'linear',0);
    wghRx	= txApod(:,:,iTx).*rxApod;
    
    % modulate if iq signal is used
    if proc.ddc.iqEnable
        rfRx	= rfRx.*exp(1i*2*pi*acq.tx.frequency*delTot);
    end
    
    % calculate rf and weights for single tx
    rfTx(:,:,iTx)	= sum(rfRx.*wghRx,3);
    wghTx(:,:,iTx)	= sum(wghRx,3);
    
    waitbar(iTx/nTx,wb);
end
close(wb);

% calculate the final rf image
rfBfr	= sum(rfTx,3)./sum(wghTx,3);

end

