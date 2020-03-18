% scan convertion
function[rfBfrOut] = scanConvert(rfBfrIn,sys,acq,proc)

[nSamp,nTx]	= size(rfBfrIn);

fs          = acq.rx.samplingFrequency/proc.ddc.decimation;
rVec        = (0:(nSamp-1))'/fs * acq.speedOfSound/2;

if acq.tx.angle==0
    xGridLin = (-(sys.nElements-1)/2:(sys.nElements-1)/2)*sys.pitch;
    zGridLin = rVec;
else
    xGridLin = (-(sys.nElements-1)/2:(sys.nElements-1)/2)*sys.pitch + rVec*sin(acq.tx.angle);
    zGridLin = repmat(rVec*cos(acq.tx.angle),[1 nTx]);
end

rfBfrOut = interp2(xGridLin,zGridLin,rfBfrIn,proc.das.xGrid,proc.das.zGrid','linear',0);

end