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
% focDel, firstSampShift
fstSampShift = 0;	% [samp] number of the sample that reflects the tx start

[nSamp,nRx,nTx] = size(rfRaw);
zSize	= length(proc.das.zGrid);
xSize	= length(proc.das.xGrid);

xElem	= (-(sys.nElements-1)/2:(sys.nElements-1)/2)*sys.pitch;

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

%% Delay & Sum
% make last samples = 0. If the algorithm needs a sample which is out of range 1:(nSamp-1), 
% then it uses the sample no. nSamp which is 0.
% Option of appending zeros as last samples instead of making last sample equal zero is slower.
rfRaw(end,:,:)	= 0;

rfTx	= zeros(zSize,xSize,nTx);
wghTx	= zeros(zSize,xSize,nTx);

wb = waitbar(0,'rf image reconstruction');
for iTx=1:nTx
    
    % calculate tx delays and apodization
    switch acq.mode
        case 'sta'
            % synthetic transmit aperture method
            txDist	= sqrt((proc.das.zGrid' - acq.tx.focus).^2 + (proc.das.xGrid-xElem(iTx)).^2);
            txDist	= txDist.*sign(proc.das.zGrid' - acq.tx.focus) + acq.tx.focus;          % WARNING: sign()=0 => invalid txDist value
            
            fNum	= abs((proc.das.xGrid-xElem(iTx))) ./ max(abs(proc.das.zGrid' - acq.tx.focus),1e-12);
            txApod	= double(fNum < 0.5);
            
        case 'pwi'
            % plane wave imaging method
            if acq.tx.angle(iTx) >= 0
                eFst	= 1;
            else
                eFst	= sys.nElements;
            end
            txDist	= (proc.das.xGrid-xElem(eFst))*sin(acq.tx.angle(iTx)) + proc.das.zGrid'*cos(acq.tx.angle(iTx));
            
            r1      = (proc.das.xGrid-xElem(   1))*cos(acq.tx.angle(iTx)) - proc.das.zGrid'*sin(acq.tx.angle(iTx));
            r2      = (proc.das.xGrid-xElem( end))*cos(acq.tx.angle(iTx)) - proc.das.zGrid'*sin(acq.tx.angle(iTx));
            txApod	= double(r1 >= 0 & r2 <= 0);
            
    end
    
    rfRx	= zeros(zSize,xSize,nRx);
    wghRx	= zeros(zSize,xSize,nRx);
    for iRx=1:nRx
        % calculate rx delays and apodization
        rxDist	= sqrt((proc.das.xGrid-xElem(iRx)).^2 + proc.das.zGrid'.^2);
        fNum	=  abs((proc.das.xGrid-xElem(iRx))./proc.das.zGrid');
        rxApod	= fNum < 0.5;
        
        % calculate total delays
        delTot	= (txDist + rxDist)/proc.speedOfSound + initDel;	% [s]
        
        % calculate sample numbers to be used in reconstruction (out-of-range sample numbers -> nSamp+1 -> sample=0)
        iSamp	= delTot*fs + 1;                 % [samp]
        iSamp(iSamp<1 | iSamp>nSamp-1) = nSamp;
        
        % calculate the rf samples (interpolated) and apodization weights
        rfRawLine       = rfRaw(:,iRx,iTx);
        rfRx(:,:,iRx)	= rfRawLine(floor(iSamp)).*(1-mod(iSamp,1)) ...
                            + rfRawLine( ceil(iSamp)).*(  mod(iSamp,1));
        wghRx(:,:,iRx)	= txApod.*rxApod;
        
        % modulate if iq signal is used
        if proc.ddc.iqEnable
            rfRx(:,:,iRx)	= rfRx(:,:,iRx).*exp(1i*2*pi*acq.tx.frequency*delTot);
        end
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

