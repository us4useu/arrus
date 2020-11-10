% Prepares rf-independent data needed for classical reconstruction
function[recPre] = reconstructRfLinPart1(sys,acq,proc)
% 
% Outputs:
% 
% recPre            - precalculated parameters needed for reconstruction:
% recPre.iSamp0     - [sample] (zSize,xSize,nRx,nTx) samples to pick
% recPre.iSamp      - [sample] (zSize,xSize,nRx,nTx) samples to pick
% recPre.modSig     - [] (zSize,xSize,nRx,nTx) re-modulation signal
% recPre.rxApod     - [] (zSize,xSize,nTx) receive weights
% recPre.iqEnable
%
% Inputs:
% 
% sys                       - system-related parameters
% sys.pitch                 - [m] probe's pitch
% 
% acq.rxSampFreq	- [Hz] sampling frequency
% acq.txFreq        - [Hz] carrier (nominal) frequency
% acq.txNPer        - [] number of periods in the emitted pulse
% acq.c             - [m/s] assumed speed of sound in the medium
% acq.txAng         - [rad] tx angle
% acq.txDelCent     - [s] (1,1) time delay between 1st rx sample and tx with the center of the tx aperture (line origin)
% 
% proc.dec          - [] decimation factor
% proc.iqEnable     - [logical] 
% proc.rxApod       - [] number of sigmas in the gaussian window used in rx apodization (0 -> rect. window)

quickRecEnable	= diff([acq.txAng; ...
                        acq.txFoc; ...
                        mod(acq.txCentElem,1); ...
                        mod(acq.rxCentElem,1); ...
                        acq.rxCentElem - acq.txCentElem],[],2) == 0;
quickRecEnable	= all(quickRecEnable(:));

fs          = acq.rxSampFreq/proc.dec;
nSamp       = acq.nSamp/proc.dec;
nRx         = acq.rxApSize;
nTx         = acq.nTx;
if quickRecEnable
    nTx0	= 1;
else
    nTx0	= nTx;
end

nSamp       = single(nSamp);
nRx         = single(nRx);
nTx0        = single(nTx0);

txAng       = reshape(acq.txAng(1:nTx0),1,1,[]);

maxTang     = tan(asin(min(1,(acq.c/acq.txFreq*2/3)/sys.pitch)));  % 2/3*Lambda/pitch -> -6dB

%% initial delays
initDel     = - acq.startSample/acq.rxSampFreq ...          % [s] rx delay with respect to start of tx
              + acq.txDelCent ...                           % [s] tx delay of the tx aperture center
              + acq.txNPer/(2*acq.txFreq);                  % [s] half the pulse length

%% Precalculate tx/rx delays and apodization
rVec        = ( (acq.startSample - 1)/acq.rxSampFreq ...
              + (0:(nSamp-1))'/fs ) * acq.c/2;              % [mm] (nSamp,1) radial distance from the line origin

xVec        = rVec.*sin(txAng);                             % [mm] (nSamp,1,1 or nTx) horiz. distance from the line origin
zVec        = rVec.*cos(txAng);                             % [mm] (nSamp,1,1 or nTx) vert.  distance from the line origin

posElem     = ((0:(nRx-1)) + acq.rxApOrig(1) - acq.rxCentElem(1))*sys.pitch;	% [mm] (1,nRx) position of the rx aperture elements along probes curvature
if isnan(sys.curvRadius)
    angElem	= zeros(1,nRx,'single');
    xElem	= posElem;
    zElem	= zeros(1,nRx,'single');
else
    angElem	= posElem / -sys.curvRadius;
    xElem	= -sys.curvRadius * sin(angElem);
    zElem	= -sys.curvRadius * (cos(angElem) - 1);
end

% warning - different definition of z=0: z is 0 for the aperture center.
% warning - different aperture position in rf simulation and reconstruction

txDist      = rVec;                                         % [mm] (nSamp,1) tx distance (from the line origin)
rxDist      = sqrt((xVec-xElem).^2 + (zVec-zElem).^2);      % [mm] (nSamp,nRx,1 or nTx) rx distance (to each rx element)

rxTang      = abs(tan(atan2(xVec-xElem,zVec-zElem) - angElem)); % [] (nSamp,nRx,1 or nTx)
rxApod      = single(rxTang < maxTang);                     % [] (nSamp,nRx,1 or nTx)
% rxApod      = single(rxTang < maxTang).*exp(-(rxTang.^2)/(2*min(1e12,maxTang/proc.rxApod)^2));
rxApod      = rxApod./sum(rxApod,2);                        % [] (nSamp,nRx,1 or nTx) normalized rx apodization vector
% warning - does the apodization takes into account for the clipped aperture?

%% Delay & Sum
t           = (txDist + rxDist)/acq.c + initDel;            % [s] (nSamp,nRx,1 or nTx) total tx-rx time delays
if proc.gpuEnable
    t       = gpuArray(t);
end

iSamp       = t*fs + 1;                                     % [samp] (nSamp,nRx,1 or nTx) sample numbers
iSamp(iSamp<1 | iSamp>nSamp)	= inf;
iSamp       = reshape(iSamp,nSamp,nRx*nTx0);

% Delay & Sum
nSamp0	= nSamp*nRx*nTx0;
if proc.gpuEnable
    nSamp0	= gpuArray(nSamp0);
end
iSamp0	= 1:nSamp0;
iSamp	= iSamp + (0:(nRx*nTx0-1))*nSamp;                   % [samp] (nSamp,nRx*nTx0)

% modulate if iq signal is used
if proc.iqEnable
    modSig	= exp(1i*2*pi*acq.txFreq*t);
end

recPre.iSamp0 = iSamp0;
recPre.iSamp  = iSamp;
recPre.modSig = modSig;
recPre.rxApod = rxApod;
recPre.iqEnable = proc.iqEnable;

end
























