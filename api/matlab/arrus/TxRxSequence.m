classdef TxRxSequence
    % Class corresponding to sequence of Tx and Rx events
    %   
    %   properties: 
    %       TxRxList - object array of class TxRx
    %
    %   methods: 
    %       TxRxSequence() - constructor. 
    %       The input should be TxRx object array (TxRxList), or nothing.
    %       Example: TxRxSeqence() creates TxRxSequence with single TxRx
    %                event of empty Tx and Rx events.
    
    properties
        TxRxList {mustBeTxRx}

        
    end
    
    methods
        function obj = TxRxSequence(varargin)                
            if nargin == 1
                obj.TxRxList = varargin{1};
                
            elseif nargin > 1 
                error('Too many arguments for TxRxSequence constructor.')

            end 

        end
        
    end
end

 
function mustBeTxRx(TxRxList)
    if ~isa(TxRxList,'TxRx') && ~isempty(TxRxList)
        error(['Bad TxRxSequence constructor input. ' ...
               'TxRxList must by the object from TxRx class.' ...
               ])
    end 
end
