% Scan convertion
function[rfBfrOut] = scanConversion(rfBfrIn,sys,acq,proc)

radGridIn	= proc.rGrid;

if sys.curvRadius == 0 && ...                   % linear/phased array
   all(diff(acq.txAng) == 0) && ...             % txAng = const
   all(diff(acq.txApCentX) > 0)                 % txApCentX increasing
    % Linear array, linear scanning, (interpolation error ~ |txAngle|)
    
    azimGridIn = acq.txApCentX;
    azimGridOut = proc.xGrid - proc.zGrid.' * tan(acq.txAng(1));
    radGridOut = repmat(proc.zGrid.' / cos(acq.txAng(1)), [1 proc.xSize]);
    
elseif sys.curvRadius == 0 || ...               % linear/phased array ...
       sys.curvRadius < 0 && ...                % or convex array
       all(diff(acq.txApCentX) == 0) && ...     % txApCent = const
       all(diff(acq.txApCentZ) == 0) && ...     % txApCent = const
       all(diff(acq.txApCentAng) == 0) && ...   % txApCent = const
       all(diff(acq.txAng) > 0)                 % txAng increasing
    % Any array (linear/phased/convex), phased scanning
    
    azimGridIn = acq.txAng + acq.txApCentAng(1);
    azimGridOut = atan2((proc.xGrid - acq.txApCentX(1)),(proc.zGrid' - acq.txApCentZ(1)));
    radGridOut = sqrt((proc.xGrid - acq.txApCentX(1)).^2 + (proc.zGrid' - acq.txApCentZ(1)).^2);
    
elseif sys.curvRadius < 0 && ...                % convex array
       all(acq.txAng == 0) && ...               % txAng = 0
       all(diff(acq.txApCentAng) > 0)           % txApCentAng increasing
    % Convex array, convex scanning
    
    azimGridIn = acq.txApCentAng;
    azimGridOut = atan2(proc.xGrid,(proc.zGrid' - sys.curvRadius - max(sys.zElem)));
    radGridOut = sqrt(proc.xGrid.^2 + (proc.zGrid' - sys.curvRadius - max(sys.zElem)).^2) + sys.curvRadius;
    
else
    error('scanConversion does not yet support the supplied probe-txApCent-txAng combination');
end

rfBfrOut = interp2(azimGridIn,radGridIn,rfBfrIn,azimGridOut,radGridOut,'linear',0);

end









