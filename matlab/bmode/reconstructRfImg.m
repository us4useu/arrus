% Reconstructs rf image from raw rf
function[rfImg] = reconstructRfImg(rf,xGrid,zGrid,pitch,fSamp,fCarr,nPer,fstSampShift,sos,txMode,txAp,txFoc,txAng)
% rfImg         - [] (zSize,xSize) output rf image

% rf            - [] (nSamp,nRx,nTx) input rf data
% xGrid         - [m] (1,xSize) x-grid vector for output rf image
% zGrid         - [m] (1,zSize) z-grid vector for output rf image
% pitch         - [m] transducer's pitch
% fSamp         - [Hz] sampling frequency
% fCarr         - [Hz] carrier frequency
% nPer          - [] number of sine periods in tx burst
% fstSampShift	- [samp] number of the sample that reflects the tx start
% sos           - [m/s] speed of sound
% txMode        - 'lin', 'sta' or 'pwi' for classical linear scan (LIN), STA scan or PWI scan
% txAp          - [elem] number of transducer elements forming the tx aperture
% txFoc         - [m] focal length for STA or LIN scheme
% txAng         - [deg] tilting angles for PWI scheme

[nSamp,nRx,nTx] = size(rf);
zSize	= length(zGrid);
xSize	= length(xGrid);

xElem	= (-(nRx-1)/2:(nRx-1)/2)*pitch;                                 % [m] x-coordinates of transducer elements

%% initial delays
fstSampDel	= fstSampShift/fSamp;                                       % [s] rx delay with respect to start of tx
burstFactor	= nPer/(2*fCarr);                                           % [s] burst factor
if any(strcmp(txMode,{'lin','sta'})) && (txFoc > 0)
    % LIN or MSTA with focusing -> need to compensate for delay of 
    focDel	= (sqrt(((txAp-1)/2*pitch).^2 + txFoc.^2) - txFoc)/sos;     % [s] focusing delay of center of tx aperture
else
    % SSTA, MSTA with defocusing, PWI -> no focusing
    focDel	= 0;
end
initDel     = - fstSampDel + focDel + burstFactor;                      % [s] total init delay

%% Delay & Sum
% add zeros as last samples. If a sample is out of range 1:nSamp, then use the sample no. nSamp+1 which is 0.
% to be checked if it is faster than irregular memory access.
rf      = [rf; zeros(1,nRx,nTx)];

rfTx	= zeros(zSize,xSize,nTx);
wghTx	= zeros(zSize,xSize,nTx);
rfRx	= zeros(zSize,xSize,nRx);
wghRx	= zeros(zSize,xSize,nRx);

wb = waitbar(0,'rf image reconstruction');
for iTx=1:nTx
    
    % calculate tx delays and apodization
    switch txMode
        case 'lin'
            % classical linear scanning (only a narrow stripe is reconstructed at a time, no tx apodization)
            xValid	= ((xGrid-xElem(iTx)) > -(pitch/2)) ...
                    & ((xGrid-xElem(iTx)) <= (pitch/2));
            nValid	= sum(xValid);
            
            txDist	= repmat(zGrid',[1 nValid]);
            txApod	= ones(zSize,nValid);
            
        case 'sta'
            % synthetic transmit aperture method
            xValid	= true(1,xSize);
            
            txDist	= sqrt((zGrid' - txFoc).^2 + (xGrid-xElem(iTx)).^2);
            txDist	= txDist.*sign(zGrid' - txFoc) + txFoc;          % WARNING: sign()=0 => invalid txDist value
            
            fNum	= abs((xGrid-xElem(iTx))) ./ max(abs(zGrid' - txFoc),1e-12);
            txApod	= double(fNum < 0.5);
            
        case 'pwi'
            % plane wave imaging method
            xValid	= true(1,xSize);
            
            if txAng(iTx) >= 0
                eFst	= 1;
            else
                eFst	= nRx;
            end
            txDist	= (xGrid-xElem(eFst))*sind(txAng(iTx)) + zGrid'*cosd(txAng(iTx));
            
            r1      = (xGrid-xElem(   1))*cosd(txAng(iTx)) - zGrid'*sind(txAng(iTx));
            r2      = (xGrid-xElem( end))*cosd(txAng(iTx)) - zGrid'*sind(txAng(iTx));
            txApod	= double(r1 >= 0 & r2 <= 0);
            
    end
    
    rfRx(:)     = 0;
    wghRx(:)	= 0;
    for iRx=1:nRx
        % calculate rx delays and apodization
        rxDist	= sqrt((xGrid(xValid)-xElem(iRx)).^2 + zGrid'.^2);
        fNum	=  abs((xGrid(xValid)-xElem(iRx))./zGrid');
        rxApod	= fNum < 0.5;
        
        % calculate total delays
        delTot	= (txDist + rxDist)/sos + initDel;	% [s]
        
        % calculate sample numbers to be used in reconstruction (out-of-range sample numbers -> nSamp+1 -> sample=0)
        iSamp	= delTot*fSamp + 1;                 % [samp]
        iSamp(iSamp<1 | iSamp>nSamp) = nSamp+1;
        
        % calculate the rf samples (interpolated) and apodization weights
        rfRawLine           = rf(:,iRx,iTx);
        rfRx(:,xValid,iRx)	= rfRawLine(floor(iSamp)).*(1-mod(iSamp,1)) ...
                            + rfRawLine( ceil(iSamp)).*(  mod(iSamp,1));
        wghRx(:,xValid,iRx)	= txApod.*rxApod;
    end
    
    % calculate rf and weights for single tx
    rfTx(:,:,iTx)	= sum(rfRx.*wghRx,3);
    wghTx(:,:,iTx)	= sum(wghRx,3);
    
    waitbar(iTx/nTx,wb);
end
close(wb);

% calculate the final rf image
rfImg	= sum(rfTx,3)./sum(wghTx,3);

end

