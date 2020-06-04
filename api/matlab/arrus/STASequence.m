classdef STASequence < SimpleTxRxSequence
    % Synthetic Transmit Aperture Tx/Rx Sequence.
    %
    
    methods
        function obj = STASequence(varargin)
            obj = obj@SimpleTxRxSequence(varargin{:});
            
        end
        
    end
end

