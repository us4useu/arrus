classdef Us4OEMSettings 
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