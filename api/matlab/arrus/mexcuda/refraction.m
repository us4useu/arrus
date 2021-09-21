function [xRefract,nIterations] = refraction(z1,x1,c1,z2,x2,c2,timePrec)

if c1>=c2
    error('Following condition must be met: c1 < c2');
end

if z1>=0 || z2<0
    error('Following conditions must be met: z1 < 0; z2 >= 0');
end

cRatio = c1 / c2;

% Initial Points
xRefractLo	= x1;                                       % initial xRefract value for c1/c2=0
sinRatioLo	= 0;

xRefractHi	= x1 + (x2-x1) * -z1 / (z2-z1);             % initial xRefract value for c1/c2=1
sinRatioHi	= 1;

timeOld     = sqrt((xRefractHi - x1)^2+z1^2)/c1 + ...
              sqrt((x2 - xRefractHi)^2+z2^2)/c2;        % timeOld = timeHi;

% Iterations
nIterations	= 0;
while true
    nIterations	= nIterations + 1;
    
    xRefractNew	= xRefractLo + (xRefractHi-xRefractLo)*(cRatio-sinRatioLo)/(sinRatioHi-sinRatioLo);
    pathLength1	= sqrt((xRefractNew - x1)^2+z1^2);
    pathLength2	= sqrt((x2 - xRefractNew)^2+z2^2);
    sinRatioNew	= ((xRefractNew - x1) / pathLength1) / ((x2 - xRefractNew) / pathLength2);
    timeNew     = pathLength1 / c1 + pathLength2 / c2;
    
    if abs(timeNew-timeOld) <= timePrec
        break;
    end
    
    if sinRatioNew < cRatio
        xRefractLo	= xRefractNew;
        sinRatioLo	= sinRatioNew;
    else
        xRefractHi	= xRefractNew;
        sinRatioHi	= sinRatioNew;
    end
    timeOld     = timeNew;
end

xRefract	= xRefractNew;
        

end
