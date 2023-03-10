classdef ProbeModelId
    % Probe model ID.
    % 
    % :param manufacturer: name of the manufacturer
    % :param name: name of the model
    properties(Constant, Hidden=true)
        REQUIRED_PARAMS = {"manufacturer", "name"};
    end

    properties
        manufacturer
        name
    end
    
    methods
        function obj = ProbeModelId(varargin)
            obj = arrus.utils.setArgs(obj, varargin, obj.REQUIRED_PARAMS);
        end
    end
end

