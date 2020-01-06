% Reconstructs rf image lines from raw rf for 'lin' mode and scanning rx aperture
function[rfBfr] = reconstructRfLin(rfRaw,sys,acq,proc,txDelCent)
% Image reconstruction: delay & sum algorithm.
% 
% Outputs:
% rfBfr         - (nSamp,nTx)  output beamformed rf
% 
% Inputs:
% rfRaw         - (nSamp,nRx,nTx) raw rf data;
%                   tx & rx apertures must be centered at the intersection of imaging line & probe surface;
%                   tx time delay of the tx aperture center element (txCentDel) must be constant for all tx's;
%                   if rfRaw is gpuArray then calculations are done on GPU;
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
rfRaw       = reshape(rfRaw,[nSamp*nRx,nTx]);

fs          = acq.rx.samplingFrequency/proc.ddc.decimation;

dT          = txDelCent + acq.tx.nPeriods/(2*acq.tx.frequency);	% [s] (1,1) total delay correction

rVec        = (0:(nSamp-1))'/fs * acq.speedOfSound/2;       % [mm] (nSamp,1) radial distance from the line origin
xVec        = rVec*sin(acq.tx.angle);                       % [mm] (nSamp,1) horiz. distance from the line origin
zVec        = rVec*cos(acq.tx.angle);                       % [mm] (nSamp,1) vert.  distance from the line origin
eVec        = (-(nRx-1)/2:(nRx-1)/2)*sys.pitch;             % [mm] (1,nElem) horiz. position of the rx aperture elements
% warning - different aperture position in rf simulation and reconstruction

txDist      = rVec;                                         % [mm] (nSamp,1) tx distance (from the line origin)
rxDist      = sqrt((xVec-eVec).^2 + zVec.^2);               % [mm] (nSamp,nRx) rx distance (to each rx element)

t           = (txDist + rxDist)/acq.speedOfSound + dT;      % [s] (nSamp,nRx) total tx-rx time delays
if isa(rfRaw,'gpuArray')
    t       = gpuArray(t);
end

spl         = t*fs + 1;                                     % [samp] (nSamp,nRx) sample numbers
spl(spl<1 | spl>nSamp-1) = inf;
spl         = reshape(spl + (0:(nRx-1))*nSamp,[],1);        % [samp] (nSamp*nRx,1)

rxFNum      = abs(xVec-eVec)./zVec;
rxApod      = rxFNum < 0.5;
rxApod      = rxApod./sum(rxApod,2);                        % [] (nSamp,nRx) normalized rx apodization vector
% warning - does the apodization takes into account for the clipped aperture?

% Delay & Sum
rfBfr	= interp1(1:(nSamp*nRx),rfRaw,spl,'linear',0);      % WARNING -> see the comment at the end of the script
rfBfr	= reshape(rfBfr,[nSamp,nRx,nTx]);

% modulate if iq signal is used
if proc.ddc.iqEnable
    rfBfr	= rfBfr.*exp(1i*2*pi*acq.tx.frequency*t);
end

rfBfr = reshape(sum(rfBfr.*rxApod,2),[nSamp,nTx]);

% WARNING
% The first argument in interp1 function is optional, code could be: rfBfr	= interp1(rfRaw,spl,'linear',0);
% However, if interp1 is executed on GPU (rfRaw is gpuArray), then interp1 contains a bug which results in CUDA error.
% The solution is to keep the first argument of the interp1 function.
% Solution found here: https://uk.mathworks.com/matlabcentral/answers/462545-interp1-gpuarray-bug

end
























