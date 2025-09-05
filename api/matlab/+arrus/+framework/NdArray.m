classdef NdArray
    % ARRUS n-dimensional array.
    %
    % :param value: matlab array to set
    % :param placement: array placement, e.g. "/Us4R:0"
    % :param name: array name, e.g. "/SequenceName/txDelays:0"
    properties(Constant, Hidden=true)
        REQUIRED_PARAMS = {'value', 'placement', 'name'};
    end

    properties
        value
        placement (1, 1)
        name (1, 1)
    end

    methods
        function obj = NdArray(varargin)
            obj = arrus.utils.setArgs(obj, varargin, obj.REQUIRED_PARAMS);
        end
    end

end
