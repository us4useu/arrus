classdef TxRxSequence
    % A sequence of Tx/Rx operations.
    %
    % :param ops: a list of TxRx operations
    % :param tgcCurve: an array of TGC samples to apply, leave empty if TGC should be turned off

    properties
        ops
        tgcCurve
    end
    
    methods
        function obj = TxRxSequence(varargin)
            p = inputParser;
            addRequired(p, 'ops', @(x) isvector(x) && ~isempty(x) && isa(x, "arrus.ops.us4r.TxRx"));
            addRequired(p, 'tgcCurve', @(x) isvector(x));
            parse(p, varargin{:});

            obj.ops = p.Results.ops;
            obj.tgcCurve = p.Results.tgcCurve;

        end
        
    end
end
