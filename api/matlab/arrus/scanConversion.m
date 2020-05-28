% Scan convertion
function[rfBfrOut] = scanConversion(rfBfrIn,sys,acq,proc)

[nSamp,nTx]	= size(rfBfrIn);

fs          = acq.rxSampFreq/proc.dec;
rVec        = ( (acq.startSample - 1)/acq.rxSampFreq ...
              + (0:(nSamp-1))'/fs ) * acq.c/2;

if acq.txAng==0
    xGridLin = sys.xElem;
    zGridLin = rVec;
else
    xGridLin = sys.xElem + rVec*sin(acq.txAng);
    zGridLin = repmat(rVec*cos(acq.txAng),[1 nTx]);
end

rfBfrOut = interp2(xGridLin,zGridLin,rfBfrIn,proc.xGrid,proc.zGrid','linear',0);

end