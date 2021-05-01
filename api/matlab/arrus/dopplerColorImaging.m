% Calculates Color/Vector Doppler Image based on set of iq images
function[color,power] = dopplerColorImaging(iqImgSet,seq,proc)
% 
% Outputs:
% color                 - (nZPix,nXPix) [rad/pri] color flow map
% power                 - (nZPix,nXPix) [dB] power flow map
% 
% Inputs:
% iqImgSet              - (nZPix,nXPix,nRep,nProj) sequence of iq images
% seq                   - structure containing sequence parameters
% seq.txAng             - [rad] tx angles
% proc                  - structure containing processing parameters
% proc.wcFiltB          - WC filter numerator
% proc.wcFiltA          - WC filter denominator
% proc.wcFiltInitCoeff  - WC filter state for step=1 initialization 
% proc.wcFiltInitSize   - number of WC filter output samples to be rejected
% proc.vectorEnable     - Vector Doppler enable
% proc.vect0Frames      - frames used for Vector Doppler reconstruction (1st projection)
% proc.vect1Frames      - frames used for Vector Doppler reconstruction (2nd projection)
% proc.vect0RxTangLim	- rx angle tangent limits for Vector Doppler reconstruction (1st projection)
% proc.vect1RxTangLim	- rx angle tangent limits for Vector Doppler reconstruction (2nd projection)

[nZPix,nXPix,nRep,nProj] = size(iqImgSet);

if ~any(nProj == [1 2])
    error('Invalid number of projections (nProj) for Color/Vector Doppler processing, nProj must equal 1 or 2.');
end

if nRep-proc.wcFiltInitSize < 2
    error('Not enough data for Color Doppler. Possibly nRep to small or wcFiltInitSize to large.');
end

%% Wall Clutter Filtration
wcFiltInitState = proc.wcFiltInitCoeff.*reshape(double(iqImgSet(:,:,1,:)), [1,nZPix,nXPix,nProj]);
iqImgSetFlt = single(filter(proc.wcFiltB, proc.wcFiltA, double(iqImgSet), wcFiltInitState, 3));

%% Mean frequency estimator (in fact - it's a mean phase shift estimator)
color = zeros(nZPix,nXPix,1,nProj,'single','gpuArray');
power = zeros(nZPix,nXPix,1,nProj,'single','gpuArray');
for iProj=1:nProj
    [color(:,:,1,iProj),power(:,:,1,iProj)] = dopplerColor(iqImgSetFlt(:,:,(proc.wcFiltInitSize+1):end,iProj));
end

%% Vector Doppler (optional)
powerDropLim = 3;   % [dB] if power of one projection is higher than the power of the other projection by more 
                    % than powerDropLim, then the weaker projection is neglected in Vector reconstruction
if proc.vectorEnable
    projMask = all(power>0,4) & (10*log10(power ./ max(power,[],4)) >= -powerDropLim);
    
    txAng	= seq.txAng([proc.vect0Frames(1) proc.vect1Frames(1)]);
    rxAng	= atan([mean(proc.vect0RxTangLim), ...
                    mean(proc.vect1RxTangLim)]);                                                                            
    txrxAng	= reshape((txAng + rxAng)/2,1,1,1,[]);                          % [rad] (1 x 1 x 1 x 2)
    
    color	= cat(4,  diff(-color.*projMask./cos(txrxAng),                    [],4) / diff(tan(txrxAng)), ...   % x-color
                    - diff(-color.*projMask./cos(txrxAng).*tan(flip(txrxAng)),[],4) / diff(tan(txrxAng)) );     % z-color
    power	= max(power.*projMask,[],4);
end

end













