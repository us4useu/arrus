% Scan convertion
function[rfBfrOut] = scanConversion(rfBfrIn,acq,proc)

nSamp       = size(rfBfrIn,1);

fs          = acq.rxSampFreq/proc.dec;
rVec        = ( (acq.startSample - 1)/acq.rxSampFreq ...
              + (0:(nSamp-1))'/fs ) * acq.c/2;

xGridLin = acq.txApCent + rVec.*sin(acq.txAng);
zGridLin = rVec.*cos(acq.txAng);

rfBfrOut = interp2(xGridLin,zGridLin,rfBfrIn,proc.xGrid,proc.zGrid','linear',0);

end