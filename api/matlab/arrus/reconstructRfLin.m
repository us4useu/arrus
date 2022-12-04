% Reconstructs rf image lines from raw rf for 'lin' mode and scanning rx aperture
function[rfBfr] = reconstructRfLin(rfRaw,sys,acq,proc)
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
% sys               - system-related parameters
% sys.pitch         - [m] probe's pitch
% sys.curvRadius    - [m] radius of probe curvature
% 
% acq               - acquisition-related parameters
% acq.rxSampFreq	- [Hz] sampling frequency
% acq.txFreq        - [Hz] carrier (nominal) frequency
% acq.initDel       - [s] initial delay due to txDelays, rxDelay, and burst factor
% acq.txAng         - [rad] tx angle
% acq.txFoc         - [rad] tx focal distance
% acq.txCentElem    - [elem] tx aperture center element
% acq.rxCentElem    - [elem] rx aperture center element
% acq.rxApOrig      - [elem] rx aperture origin element
% acq.startSample   - [samp] starting sample number
% acq.txDelCent     - [s] (1,1) time delay between 1st rx sample and tx with the center of the tx aperture (line origin)
% acq.hwDdcEnable   - [logical] hardware DDC enable
% 
% proc              - processing-related parameters
% proc.swDdcEnable  - [logical] software DDC enable
% proc.dec          - [] software DDC decimation factor
% proc.sos          - [m/s] assumed speed of sound in the medium
% proc.rxApod       - [] number of sigmas in the gaussian window used in rx apodization (0 -> rect. window)

quickRecEnable	= diff([acq.txAng; ...
                        acq.txFoc; ...
                        mod(acq.txCentElem,1); ...
                        mod(acq.rxCentElem,1); ...
                        acq.rxCentElem - acq.txCentElem],[],2) == 0;
quickRecEnable	= all(quickRecEnable(:));

if quickRecEnable
    txAng	= acq.txAng(1);
    txFreq	= acq.txFreq(1);
    dT      = acq.initDel(1);
else
    txAng	= reshape(acq.txAng,1,1,[]);
    txFreq	= reshape(acq.txFreq,1,1,[]);
    dT      = reshape(acq.initDel,1,1,[]);
end

%% Reconstruction
[nSamp,nRx,nTx]	= size(rfRaw);
rfRaw       = reshape(rfRaw,[nSamp*nRx,nTx]);

fs          = acq.rxSampFreq/proc.dec;

maxTang     = tan(asin(min(1,(proc.sos./txFreq*2/3)/sys.pitch)));  % 2/3*Lambda/pitch -> -6dB

rVec        = ( (acq.startSample - 1)/acq.rxSampFreq ...
              + (0:(nSamp-1))'/fs ) * proc.sos/2;           % [mm] (nSamp,1) radial distance from the line origin

xVec        = rVec.*sin(txAng);                             % [mm] (nSamp,1,1 or nTx) horiz. distance from the line origin
zVec        = rVec.*cos(txAng);                             % [mm] (nSamp,1,1 or nTx) vert.  distance from the line origin

posElem     = ((0:(nRx-1)) + acq.rxApOrig(1) - acq.rxCentElem(1))*sys.pitch;	% [mm] (1,nRx) position of the rx aperture elements along probes curvature
if isnan(sys.curvRadius)
    angElem	= zeros(1,nRx);
    xElem	= posElem;
    zElem	= zeros(1,nRx);
else
    angElem	= posElem / -sys.curvRadius;
    xElem	= -sys.curvRadius * sin(angElem);
    zElem	= -sys.curvRadius * (cos(angElem) - 1);
end

% warning - different definition of z=0: z is 0 for the aperture center.
% warning - different aperture position in rf simulation and reconstruction

txDist      = rVec;                                         % [mm] (nSamp,1) tx distance (from the line origin)
rxDist      = sqrt((xVec-xElem).^2 + (zVec-zElem).^2);      % [mm] (nSamp,nRx,1 or nTx) rx distance (to each rx element)

t           = (txDist + rxDist)/proc.sos + dT;              % [s] (nSamp,nRx,1 or nTx) total tx-rx time delays
if isa(rfRaw,'gpuArray')
    t       = gpuArray(t);
end

iSamp       = t*fs + 1;                                     % [samp] (nSamp,nRx,1 or nTx) sample numbers
iSamp(iSamp<1 | iSamp>nSamp)	= inf;
iSamp       = reshape(iSamp + (0:(nRx-1))*nSamp,nSamp*nRx,[]);	% [samp] (nSamp*nRx,1 or nTx)

rxTang      = abs(tan(atan2(xVec-xElem,zVec-zElem) - angElem)); % [] (nSamp,nRx,1 or nTx)
rxApod      = double(rxTang < maxTang);                     % [] (nSamp,nRx,1 or nTx)
% rxApod      = double(rxTang < maxTang).*exp(-(rxTang.^2)/(2*min(1e12,maxTang/proc.rxApod)^2));
rxApod      = rxApod./sum(rxApod,2);                        % [] (nSamp,nRx,1 or nTx) normalized rx apodization vector
% warning - does the apodization takes into account for the clipped aperture?

% Delay & Sum
if quickRecEnable
    rfBfr	= interp1(1:(nSamp*nRx),rfRaw,iSamp,'linear',0);	% WARNING -> see the comment at the end of the script
else
    rfBfr	= interp2(1:nTx,(1:(nSamp*nRx)).',rfRaw,(1:nTx).*ones(nSamp*nRx,1),iSamp,'linear',0);
end
rfBfr	= reshape(rfBfr,[nSamp,nRx,nTx]);

% modulate if iq signal is used
if acq.hwDdcEnable || proc.swDdcEnable
    rfBfr	= rfBfr.*exp(1i*2*pi*txFreq.*t);
end

rfBfr = reshape(sum(rfBfr.*rxApod,2),[nSamp,nTx]);

% WARNING
% The first argument in interp1 function is optional, code could be: rfBfr	= interp1(rfRaw,iSamp,'linear',0);
% However, if interp1 is executed on GPU (rfRaw is gpuArray), then interp1 contains a bug which results in CUDA error.
% The solution is to keep the first argument of the interp1 function.
% Solution found here: https://uk.mathworks.com/matlabcentral/answers/462545-interp1-gpuarray-bug

end
























