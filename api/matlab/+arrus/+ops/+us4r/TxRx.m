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

    properties
        tx
        rx
        pri
    end

    methods
        function obj = TxRx(varargin)
            % TxRx constructor.
            % To pass arguments to the constructor name-value convetion is used.
            %


            p = inputParser;
            addRequired(p, 'tx', @(x) isa(x, 'arrus.ops.us4r.Tx'));
            addRequired(p, 'rx', @(x) isa(x, 'arrus.ops.us4r.Rx'));
            addRequired(p, 'pri', @(x) isreal(x) && isscalar(x) && x > 0);
            parse(p, varargin{:});
                                
            obj.tx = p.Results.tx;
            obj.rx = p.Results.rx;
            obj.pri = p.Results.pri;
        end
        
        
    end
end
