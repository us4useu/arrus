classdef ProbeModel
    % Probe model.
    % 
    % :param modelId: id of the model
    % :param nElements: (scalar (for 2-D probe) or a pair (for 3-D probe))\
    %   probe's number of elements 
    % :param pitch: (scalar (for 2-D probe) or a pair (for 3-D probe))\
    %   probe's element pitch
    % :param txFrequencyRange: (a pair - two-element vector) 
    %   a range [min, max] of the available tx center frequencies  
    
    properties(GetAccess = public, SetAccess = private)
        modelId arrus.devices.probe.ProbeModelId
        nElements 
        pitch
        txFrequencyRange (1, 2)
    end
    
    methods
        function obj = ProbeModel(modelId, nElements, pitch, ...
                     txFrequencyRange)
            obj.modelId = modelId;
            obj.nElements = nElements;
            obj.pitch = pitch;
            obj.txFrequencyRange = txFrequencyRange;
        end
    end
    
end