function probe = probeParams(probeName)

%% Probe parameters
switch probeName
    case 'AL2442'
        probe.nElem	= 192;
        probe.pitch	= 0.21e-3;
        
    case 'SL1543'
        probe.nElem	= 192;
        probe.pitch	= 0.245e-3;
        
%     case 'SP2430'
%         probe.nElem	= 96;
%         probe.pitch	= 0.22e-3;
        
    case 'AC2541'
        probe.nElem	= 192;
        probe.pitch	= 0.30e-3;
        
    case 'L14-5/38'
        probe.nElem	= 128;
        probe.pitch	= 0.3048e-3;
        
    otherwise
        disp('Invalid probe name');
        probe = [];
        return;
        
end

%% Adapter type & channel mapping
switch probeName
    case {'AL2442','SL1543','AC2541'}
        probe.adapType      = 0;
        
        probe.rxChannelMap	= [32:-1:1; 1:1:32];
        probe.rxChannelMap(1,[16 17]) = [16 17];
        
        probe.txChannelMap	= [ probe.rxChannelMap +  0, ...
                                probe.rxChannelMap + 32, ...
                                probe.rxChannelMap + 64, ...
                                probe.rxChannelMap + 96];
    
    case 'L14-5/38'
        probe.adapType      = 1;
        
        probe.rxChannelMap	= [1:1:32; 32:-1:1];
        
        probe.txChannelMap	= [ probe.rxChannelMap +  0, ...
                                probe.rxChannelMap + 32, ...
                                probe.rxChannelMap + 64, ...
                                probe.rxChannelMap + 96];
end

end

