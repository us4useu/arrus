classdef ProbeModel
    % Probe model.
    % 
    % :param modelId: id of the model
    % :param nElements: (scalar (for 2-D probe) or a pair (for 3-D probe))\
    %   probe's number of elements 
    % :param pitch: (scalar (for 2-D probe) or a pair (for 3-D probe))\
    %   probe's element pitch [m]
    % :param txFrequencyRange: (a pair - two-element vector) 
    %   a range [min, max] of the available tx center frequencies
    % :param voltageRange: (a pair - two-element vector)
    %   a range [min, max] of the available TX voltage
    % :param curvatureRadius: probe curvature radius, equal 0 for linear arrays [m]

    properties(Constant, Hidden=true)
        REQUIRED_PARAMS = {"modelId", "nElements", "pitch", "txFrequencyRange", "voltageRange", "curvatureRadius"};
    end
    
    properties
        modelId arrus.devices.probe.ProbeModelId
        nElements 
        pitch
        txFrequencyRange
        voltageRange
        curvatureRadius
        lens
        matchingLayer
    end
    
    methods
        function obj = ProbeModel(varargin)
            obj = arrus.utils.setArgs(obj, varargin, obj.REQUIRED_PARAMS);
        end
    end
    
end