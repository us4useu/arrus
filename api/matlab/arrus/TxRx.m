classdef TxRx
    % class corresponding to single 'transmit-and-receive' event
    % 
    %   properties:
    %       Tx - Tx object
    %       Rx - Rx object
    properties
        Tx@Tx = Tx()
        Rx@Rx = Rx()
    end
    
    
    methods
        function obj = TxRx(varargin)
            if nargin ~= 0
                p = inputParser;
                                
                % adding parameters to parser
                addParameter(p, 'Tx',Tx())
                addParameter(p, 'Rx',Rx())
                parse(p,varargin{:})
                                
                obj.Tx = p.Results.Tx;
                obj.Rx = p.Results.Rx;
            end                 
        end
        
        
    end
end
