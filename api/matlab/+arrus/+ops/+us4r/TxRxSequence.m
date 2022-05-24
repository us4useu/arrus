classdef TxRxSequence
    % A sequence of Tx/Rx operations.
    %
    % :param ops: a list of TxRx operations. Required.
    % :param tgcCurve: an array of TGC samples to apply, leave empty if analog TGC should be turned off
    % :param sri: sequence repetition interval - the time between consecutive RF frames.
    %  If empty, SRI is determined by the total pri only. [s]
    % :param nRepeats: how many time this sequence should be repeated, default is 1.
    properties(Constant, Hidden=true)
        REQUIRED_PARAMS = {'ops'};
    end

    properties
        ops arrus.ops.us4r.TxRx
        tgcCurve (1, :) {mustBeReal} = []
        sri (1, 1) {mustBeReal, mustBePositive} = []
        nRepeats (1, 1) {mustBePositive, mustBeInteger} = 1
    end
    methods
        function obj = TxRxSequence(varargin)
            obj = arrus.utils.setArgs(obj, varargin, obj.REQUIRED_PARAMS);
        end
    end

end
