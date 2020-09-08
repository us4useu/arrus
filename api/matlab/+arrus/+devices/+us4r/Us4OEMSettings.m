classdef Us4OEMSettings 
    % Us4OEM module settings.
    % 
    % :param channelMapping: (128 element vector) channel mapping 
    %   (permutation) to apply for given Us4OEM
    % :param activeChannelGroups: (16 element vector) a boolean vector, 
    %   true at position `i` means that the i-th group should be active.
    % :param rxSettings: Rx settings to apply to given Us4OEM

    properties(GetAccess = public, SetAccess = private)
        channelMapping (1, 128)
        activeChannelGroups (1, 16)
        rxSettings arrus.devices.us4r.RxSettings
    end
    
    methods(Access = public)
        function obj = Us4OEMSettings(channelMapping, ... 
                                      activeChannelGroups, ...
                                      rxSettings)
                                  
            obj.channelMapping = channelMapping;
            obj.activeChannelGroups = activeChannelGroups;
            obj.rxSettings = rxSettings;
        end
    end
    
end