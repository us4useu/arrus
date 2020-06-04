classdef PWISequence < SimpleTxRxSequence
    % Plane Wave Imaging Tx/Rx Sequence.
    %
    % The PWI sequence has txFocus set to inf by defualt.
    
    methods
        function obj = PWISequence(varargin)
            obj = obj@SimpleTxRxSequence(varargin{:});
            
            % Fix the txFocus value
            if any(~isinf(obj.txFocus))
                obj.txFocus(:) = inf;
                warning("PWISequence: txFocus is forced to be all inf");
            end
            
        end
        
    end
end

