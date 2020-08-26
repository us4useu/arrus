% Scan convertion
function[rfBfrOut] = scanConversion(rfBfrIn,acq,proc)

nSamp       = size(rfBfrIn,1);
fs          = acq.rxSampFreq/proc.dec;

radGridIn	= ( (acq.startSample - 1)/acq.rxSampFreq ...
              + (0:(nSamp-1))'/fs ) * acq.c/2;

if all(diff(acq.txAng) == 0) && ...     % txAng = const
   all(diff(acq.txApCent) > 0)          % txApCent increasing
    % Linear scanning
    
    azimGridIn = acq.txApCent;
    azimGridOut = proc.xGrid - proc.zGrid.' * tan(acq.txAng(1));
    radGridOut = repmat(proc.zGrid.' / cos(acq.txAng(1)), [1 proc.xSize]);
    
elseif all(diff(acq.txApCent) == 0) && ...	% txApCent = const
       all(diff(acq.txAng) > 0)             % txAng increasing
    % Phased scanning
    
    azimGridIn = acq.txAng;
    azimGridOut = atan2(proc.xGrid - acq.txApCent(1), proc.zGrid');
    radGridOut = sqrt((proc.xGrid - acq.txApCent(1)).^2 + proc.zGrid'.^2);
    
end

rfBfrOut = interp2(azimGridIn,radGridIn,rfBfrIn,azimGridOut,radGridOut,'linear',0);

end








