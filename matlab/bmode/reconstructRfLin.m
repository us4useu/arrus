% Reconstructs rf image lines from raw rf for 'lin' mode and scanning rx aperture
function[rfLin] = reconstructRfLin(rfRaw,sys,            txDelCent,txAng,rxApert)
% Image reconstruction: delay & sum algorithm.
% 
% Outputs:
% rfLin         - [] (nSamp,nTx) rf image lines
% 
% Inputs:
% rfRaw         - [] (nSamp,nRx,nTx) raw rf data; tx & rx apertures must be centered at the intersection of imaging line
%               & probe surface; tx time delay of the tx aperture center element (txCentDel) must be constant for all tx's;
% sys           - system-related parameters
% sys.pitch     - [m] transducer pitch
% sys.fs        - [Hz] sampling frequency
% sys.fn        - [Hz] carrier (nominal) frequency
% sys.nPer      - [] number of periods in the emitted pulse
% sys.sos       - [m/s] assumed speed of sound in the medium


% txDelCent	- [s] (1,1) time delay between 1st rx sample and tx with the center of the tx aperture (line origin)
% txAng     - [deg] (1,1) tx angle
% rxApert	- [elem] rx aperture size; skip it if rfRaw is in scanning rx aperture format; 
%           if rfRaw is in full rx aperture format, the rxApert allows to select proper rx channels;


[nSamp,nRx,nTx]	= size(rfRaw);

isIq	= any(~isreal(rfRaw));

%% Convert input rf data to a format of scanning rx aperture
if nargin==5 && ~isempty(rxApert)
    rxApertCent	= 1:nTx;
    rxApertFst	= rxApertCent - floor((rxApert-1)/2);
    rxApertLst	= rxApertCent +  ceil((rxApert-1)/2);
    
    rfRawOld	= rfRaw;
    rfRaw       = zeros(nSamp,rxApert,nTx);
    for iTx=1:nTx
        rxApertOld =    max(1,  rxApertFst(iTx))  :          min(nRx,rxApertLst(iTx));
        rxApertNew = (1+max(0,1-rxApertFst(iTx))) : (rxApert-max(0,rxApertLst(iTx)-nRx));
        rfRaw(:,rxApertNew,iTx) = rfRawOld(:,rxApertOld,iTx);
    end
    
    nRx = rxApert;
end

%% Reconstruction
dT          = txDelCent + sys.nPer/(2*sys.fn);                              % [s] (1,1) total delay correction

rVec        = (0:(nSamp-1))'/sys.fs*sys.sos/2;                              % [mm] (nSamp,1) radial distance from the line origin
xVec        = rVec*sind(txAng);                                             % [mm] (nSamp,1) horiz. distance from the line origin
zVec        = rVec*cosd(txAng);                                             % [mm] (nSamp,1) vert.  distance from the line origin
eVec        = (-(nRx-1)/2:(nRx-1)/2)*sys.pitch;                             % [mm] (1,nElem) horiz. position of the rx aperture elements (rel. to the line origin)
% warning - different aperture position in rf simulation and reconstruction

txDist      = rVec;                                                         % [mm] (nSamp,1) tx distance (from the line origin to each point of the imaging line)
rxDist      = sqrt((xVec-eVec).^2 + zVec.^2);                               % [mm] (nSamp,nRx) rx distance (from each point of the imaging line to each element of the tx aperture)

rfRaw(end,:,:)	= 0;                                                        % zero the last sample to efficiently skip out-of-range samples in later processing

t           = (txDist + rxDist)/sys.sos + dT;                               % [s] (nSamp,nRx) total tx-rx time delays

spl         = t*sys.fs + 1;                                                 % [samp] (nSamp,nRx) sample numbers to be used in image line reconstruction
spl(spl<1 | spl>nSamp-1) = nSamp;


splPrev     = floor(spl);                                                   % [samp] (nSamp,nRx) previous sample numbers (for interpolation, if the spl isn't integer)
splNext     = ceil(spl);                                                    % [samp] (nSamp,nRx) next sample numbers (for interpolation, if the spl isn't integer)

fNum        = abs(xVec-eVec)./zVec;
rxApod      = fNum < 0.5;
rxApod      = permute(rxApod./sum(rxApod,2),[1 3 2]);                       % [] (nSamp,1,nRx) normalized & reorganized rx apodization vector
% warning - does the apodization takes into account for the clipped aperture?

% Delay & Sum
rfLin       = zeros(nSamp,nTx,nRx);
for iRx=1:nRx
	rfLin(:,:,iRx)	= squeeze(rfRaw(splPrev(:,iRx),iRx,:)).*(1-mod(spl(:,iRx),1)) + ...
                      squeeze(rfRaw(splNext(:,iRx),iRx,:)).*(  mod(spl(:,iRx),1));
end

% modulate if iq signal is used
if isIq
    rfLin	= rfLin.*exp(1i*2*pi*sys.fn*permute(t,[1 3 2]));
end

rfLin = sum(rfLin.*rxApod,3);

% Optimization:
% weighting by modulo - slower by ~10ms (470->480ms) but code is nicer than splNext-spl or spl-splPrev;
% permute(rfRaw,[1 3 2]) and avoid squeezeng - cost of 'permute' is higher than gain from avoiding 'squeeze';

end
























