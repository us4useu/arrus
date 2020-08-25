classdef TxRx
    % Class corresponding to single 'transmit-and-receive' event
    % 
    %   properties:
    %       Tx - Tx object
    %       Rx - Rx object
    %       pri - pulse repetition interval [s]    
    %
    %   methods:
    %       TxRx() - constructor.
    %           TxRx() creates Rx object with all empty properties.
    %           To pass arguments to the constructor name-value convetion is used.
    %           Example: Rx('rxAperture', logical(1:128))    
    
    
    
    properties
        Tx@Tx
        Rx@Rx
        pri
    end
    
    
    methods
        function obj = TxRx(varargin)
            if nargin ~= 0
                p = inputParser;
                                
                priVld = @(x) isreal(x) && isscalar(x) && x > 0 ...
                    || isequal(x,'min');
                % adding parameters to parser
                addParameter(p, 'Tx', Tx())
                addParameter(p, 'Rx', Rx())
                addParameter(p, 'pri', 2e-3, priVld)
                parse(p, varargin{:})
                                
                obj.Tx = p.Results.Tx;
                obj.Rx = p.Results.Rx;
                obj.pri = p.Results.pri;
            end                 
        end
        
        
    end
end
