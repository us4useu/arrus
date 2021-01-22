classdef ProbeSettings
    % Probe adapter settings.
    %
    % :param probeModel: probe's model description
    % :param channelMapping: (vector 1 x nChannels) channel mapping to \
    %   apply; if the `i`-th value is equal to `j`, it means that the \
    %   probe's channel `i` is connected to connector's channel `j`.
    
    properties(GetAccess = public, SetAccess = private)
        probeModel arrus.devices.probe.ProbeModel
        channelMapping
    end
    
    methods(Access = public)
        
        function obj = ProbeSettings(probeModel, channelMapping)
            obj.probeModel = probeModel;
            obj.channelMapping = channelMapping;
        end
        
    end
end