classdef CustomTxRxSequence < SimpleTxRxSequence
    % Custom Tx/Rx Sequence.
    %
    % CustomTxRxSequence will replace SimpleTxRxSequence in the future.
    
    methods
        function obj = CustomTxRxSequence(varargin)
            obj = obj@SimpleTxRxSequence(varargin{:});
        end
        
    end
end

