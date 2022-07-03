classdef DataBufferDef
    % Class describing output data buffer properties.
    %
    % :param bufferType: buffer type, currently available values: FIFO
    % :param nElements: number of elements the buffer should contain
    properties(Constant, Hidden=true)
        REQUIRED_PARAMS = {};
    end
    properties
        type (1, 1) = "FIFO"
        nElements (1, 1) {mustBeFinite, mustBeReal, mustBePositive} = 2
    end

    methods
        function obj = DataBufferDef(varargin)
            obj = arrus.utils.setArgs(obj, varargin, obj.REQUIRED_PARAMS);
        end
    end
end
