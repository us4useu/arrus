classdef Scheme
    % A scheme to be executed within the session.
    % Note: the number of elements refers to the number of batches that should be allocated
    % on device or host memory.
    %
    % :param txRxSequence: TX/RX sequence to be used in scheme
    % :param rxBufferSize: the size of the buffer allocated on the ultrasound device [number of elements]
    % :param outputBuffer: definition of the ultrasound device output buffer on host
    % :param workMode: ultrasound device work mode, available: "HOST", "ASYNC", "MANUAL"
    properties(Constant, Hidden=true)
        REQUIRED_PARAMS = {'txRxSequence'};
    end
    properties
        txRxSequence arrus.ops.us4r.TxRxSequence
        rxBufferSize (1, 1) {mustBeFinite, mustBeReal, mustBePositive} = 2
        outputBuffer arrus.framework.DataBufferDef = arrus.framework.DataBufferDef("type", "FIFO", "nElements", 2)
        workMode (1, 1) = "HOST"
    end
    methods
        function obj = Scheme(varargin)
            obj = arrus.utils.setArgs(obj, varargin, obj.REQUIRED_PARAMS);
        end
    end
end
