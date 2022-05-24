classdef Scheme
    % A scheme to be executed within the session.
    %
    % :param ops: a list of TxRx operations
    % :param tgcCurve: an array of TGC samples to apply, leave empty if TGC should be turned off
    properties
        txRxSequence arrus.ops.us4r.TxRxSequence
        rxBufferSize (1, 1) {mustBeFinite, mustBeReal, mustBePositive} = 2
        outputBuffer arrus.framework.DataBufferDef = arrus.framework.DataBufferDef("FIFO", 2)
        workMode (1, 1) {mustBeStringScalar} = "HOST"
    end
end
