classdef LINSequence < SimpleTxRxSequence
   % Classical imaging Tx/Rx Sequence.
    %
    
    methods
        function obj = LINSequence(varargin)
            obj = obj@SimpleTxRxSequence(varargin{:});
            
        end
        
    end
end

