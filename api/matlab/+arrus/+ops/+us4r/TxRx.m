classdef TxRx
    % A class encapsulating a single 'transmit-and-receive' op.
    %
    % Example usage:
    %
    % .. code:: matlab
    %
    %   TxRx('tx', tx, 'rx', rx, 'pri', 100e-6)
    % 
    % :param tx: pulse emission, object of type `arrus.ops.us4r.Tx`
    % :param rx: echo data reception, object of type `arrus.ops.us4r.Rx`
    % :param pri: pulse repetition interval [s]
    properties(Constant, Hidden=true)
        REQUIRED_PARAMS = {'tx', 'rx', 'pri'};
    end

    properties
        tx  arrus.ops.us4r.Tx {arrus.validators.mustBeSingleObject}
        rx  arrus.ops.us4r.Rx {arrus.validators.mustBeSingleObject}
        pri {mustBeFinite, mustBeReal, mustBePositive} = 1e6
    end

    methods
        function obj = TxRx(varargin)
            obj = arrus.utils.setArgs(obj, varargin, obj.REQUIRED_PARAMS);
        end
    end
end
