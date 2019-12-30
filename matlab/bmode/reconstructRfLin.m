% Reconstructs rf image lines from raw rf for 'lin' mode and scanning rx aperture
function[rfBfr] = reconstructRfLin(rfRaw,sys,acq,proc,txDelCent)
% Image reconstruction: delay & sum algorithm.
% 
% Outputs:
% rfBfr         - (nSamp,nTx)  output beamformed rf
% 
% Inputs:
% rfRaw         - (nSamp,nRx,nTx) raw rf data; tx & rx apertures must be centered at the intersection of imaging line
%               & probe surface; tx time delay of the tx aperture center element (txCentDel) must be constant for all tx's;
%               if rfRaw is gpuArray then calculations are done on GPU;
% 
% sys                       - system-related parameters
% sys.nElements             - [elem] number of probe's elements
% sys.pitch                 - [m] probe's pitch
% 
% acq.rx.samplingFrequency	- [Hz] sampling frequency
% acq.tx.frequency          - [Hz] carrier (nominal) frequency
% acq.tx.nPeriods           - [] number of periods in the emitted pulse
% acq.speedOfSound          - [m/s] assumed speed of sound in the medium
% acq.tx.angle              - [rad] tx angle
% 
% 
% proc.ddc.decimation       - [] decimation factor
% proc.ddc.iqEnable         - [logical] 
% proc.das.xGrid            - 
% proc.das.zGrid            - 

% txDelCent	- [s] (1,1) time delay between 1st rx sample and tx with the center of the tx aperture (line origin)

%% Reconstruction
[nSamp,nRx,nTx]	= size(rfRaw);

fs          = acq.rx.samplingFrequency/proc.ddc.decimation;

dT          = txDelCent + acq.tx.nPeriods/(2*acq.tx.frequency);	% [s] (1,1) total delay correction

rVec        = (0:(nSamp-1))'/fs * acq.speedOfSound/2;           % [mm] (nSamp,1) radial distance from the line origin
xVec        = rVec*sin(acq.tx.angle);                           % [mm] (nSamp,1) horiz. distance from the line origin
zVec        = rVec*cos(acq.tx.angle);                           % [mm] (nSamp,1) vert.  distance from the line origin
eVec        = (-(nRx-1)/2:(nRx-1)/2)*sys.pitch;                 % [mm] (1,nElem) horiz. position of the rx aperture elements (rel. to the line origin)
% warning - different aperture position in rf simulation and reconstruction

txDist      = rVec;                                                         % [mm] (nSamp,1) tx distance (from the line origin to each point of the imaging line)
rxDist      = sqrt((xVec-eVec).^2 + zVec.^2);                               % [mm] (nSamp,nRx) rx distance (from each point of the imaging line to each element of the tx aperture)

rfRaw(end,:,:)	= 0;                                                        % zero the last sample to efficiently skip out-of-range samples in later processing

t           = (txDist + rxDist)/acq.speedOfSound + dT;                               % [s] (nSamp,nRx) total tx-rx time delays

spl         = t*fs + 1;                                                 % [samp] (nSamp,nRx) sample numbers to be used in image line reconstruction
spl(spl<1 | spl>nSamp-1) = nSamp;


splPrev     = floor(spl);                                                   % [samp] (nSamp,nRx) previous sample numbers (for interpolation, if the spl isn't integer)
splNext     = ceil(spl);                                                    % [samp] (nSamp,nRx) next sample numbers (for interpolation, if the spl isn't integer)

fNum        = abs(xVec-eVec)./zVec;
rxApod      = fNum < 0.5;
rxApod      = permute(rxApod./sum(rxApod,2),[1 3 2]);                       % [] (nSamp,1,nRx) normalized & reorganized rx apodization vector
% warning - does the apodization takes into account for the clipped aperture?

% % Delay & Sum
rfRaw	= reshape(rfRaw,[nSamp*nRx,nTx]);
splPrev	= reshape(splPrev + (0:(nRx-1))*nSamp,[],1);
splNext	= reshape(splNext + (0:(nRx-1))*nSamp,[],1);
spl     = spl(:);
rfBfr	= rfRaw(splPrev,:).*(1-mod(spl,1)) ...
        + rfRaw(splNext,:).*   mod(spl,1);
rfBfr	= permute(reshape(rfBfr,[nSamp,nRx,nTx]),[1 3 2]);

% modulate if iq signal is used
if proc.ddc.iqEnable
    rfBfr	= rfBfr.*exp(1i*2*pi*acq.tx.frequency*permute(t,[1 3 2]));
end

rfBfr = sum(rfBfr.*rxApod,3);

% Optimization:
% weighting by modulo - slower by ~10ms (470->480ms) but code is nicer than splNext-spl or spl-splPrev;
% permute(rfRaw,[1 3 2]) and avoid squeezeng - cost of 'permute' is higher than gain from avoiding 'squeeze';

end
























