classdef TxRxSequence
    % class corresponding to sequence of Tx and Rx events
    %   
    %   properties: 
    %       TxRxList - object array of class TxRx
    %
    %   methods: 
    %       TxRxSequence() - constructor. The input should be TxRx object
    %       array (TxRxList).
    
    properties
        TxRxList {mustBeTxRx} = TxRx()
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
    if ~isa(TxRxList,'TxRx')
        error(['Bad TxRxSequence constructor input. ' ...
               'TxRxList must by the object from TxRx class.' ...
               ])
    end 
end
