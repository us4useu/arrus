% function LINSequence(aperture, focus, probe, c)
%     % TODO inputs: tx aperture size, etc., probe (arrus.devices.probe.Probe), assumed speed of sound / Medium object
%     % TODO output: a sequence of tx rx operations to apply on a given probe
%     
%     
%     
%     
% end


function txrxlist = LINSequence(nElem, nElemSub, fc,pitch, c, focalDepth, sampleRange,pri)
    
    import arrus.ops.us4r.*
    
    if mod(nElemSub,2)
        focus = [-pitch/2, focalDepth];
    else
        focus = [0,focalDepth];
    end
    
    nPeriods = 2;
    pulse = Pulse(fc, nPeriods);
    subApertureDelays = enumClassicDelays(nElemSub,pitch, c, focus);
    [apMasks,apDelays] = simpleApertureScan(nElem, subApertureDelays);
    
    for iElem = 1:nElem
        thisAp = apMasks(iElem,:);
        thisDel = apDelays(iElem,:);
        tx = Tx(thisAp,thisDel,pulse);
        rx = Rx(thisAp,sampleRange);
        txrx = TxRx(tx,rx,pri);
        txrxlist(iElem) = txrx;
    end
    
end


function [apMasks,apDelays] = simpleApertureScan(nElem, subApertureDelays)
% Function generate array which describes which elements are turn on during
% 'classic' txrx scheme scan. The supaberture step is equal 1.  
% 
% :param nElem: number of elements in the array, i.e. full aperture length,
% :param subApLen: subaperture length,
% :param apMasks: output array of subsequent aperture masks 
%           (nLines x nElements size),
% :param apDelays: output array of subsequent aperture delays
%           (nLines x nElements size).

    subApLen = length(subApertureDelays);
    bigHalf = ceil(subApLen/2);
    smallHalf = floor(subApLen/2);
    apMasks = false(nElem, nElem);
    apDelays = zeros(nElem, nElem);
    for iElement = 1:nElem
        vAperture = false(1, smallHalf + nElem + bigHalf);
        vAperture(1, iElement:iElement+subApLen-1) = true;
        apMasks(iElement, :) = vAperture(bigHalf:end - smallHalf-1);
        
        vDelays = zeros(1, smallHalf + nElem + bigHalf);
        vDelays(1, iElement:iElement+subApLen-1) = subApertureDelays;
        apDelays(iElement, :) = vDelays(bigHalf:end - smallHalf-1);
        
    end
end


function delays = enumClassicDelays(nElem, pitch, c, focus)
% The function enumerates classical focusing delays for linear array. 
% It assumes that the 0 is in the center of the aperture.
%
% :param nElem: number of elements in aperture,
% :param pitch: distance between two adjacent probe elements [m],
% :param c: speed of sound [m/s],
% :param focus: coordinates of the focal point ([xf,zf]), 
%               or focal length only (then xf = 0 is assumed) [m],
% :param delays: output delays vector.


    if isscalar(focus)
        xf = 0;
        zf = focus;
    elseif length(focus(:)) == 2
        xf = focus(1);
        zf = focus(2);
    else
        error('Inproper focus value.')
    end

    probeWidth = (nElem-1)*pitch;
    elCoordX = linspace(-probeWidth/2, probeWidth/2, nElem);
    element2FocusDistance = sqrt((elCoordX-xf).^2 + zf.^2);
    distMax = max(element2FocusDistance);
    delays = (distMax - element2FocusDistance)/c;

end
