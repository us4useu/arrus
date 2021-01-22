classdef ProbeAdapterSettings
    % Probe adapter settings.
    %
    % :param modelId: id of the model 
    % :param nChannels: number of adapter output channels
    % :param channelMapping: (matrix 2 x nChannels) channel mapping to \
    %   apply; for each i-th column the first row should be equal to the \
    %   Us4OEM's ordinal number (`o`), second row should be equal to \
    %   Us4OEM's channel; such a column means that adapter's channel \
    %   `i` is connected to o-th Us4OEM, channel number `ch`.

    properties(GetAccess = public, SetAccess = private)
        modelId arrus.devices.us4r.ProbeAdapterModelId
        nChannels (1, 1)
        channelMapping 
    end
    
    methods(Access = public)
        function obj = ProbeAdapterSettings(modelId, nChannels, ...
                                      channelMapping)
                                  
            obj.modelId = modelId;
            obj.nChannels = nChannels;
            obj.channelMapping = channelMapping;
        end
    end
    
end