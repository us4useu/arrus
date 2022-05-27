classdef DataBufferDef
    % Class describing output data buffer properties.
    %
    % :param bufferType: buffer type, currently available values: FIFO
    % :param nElements: number of elements the buffer should contain
    properties
        type (1, 1) {mustBeStringScalar} = "FIFO"
        nElements (1, 1) {mustBeFinite, mustBeReal, mustBePositive} = 2
    end
end
