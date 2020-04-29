classdef PWISequence < SimpleTxRxSequence
    % Plane Wave Imaging Tx/Rx Sequence.
    %
    % The PWI sequence has set txFocus to inf by defualt.
    
    methods
        function obj = PWISequence(varargin)
            obj = obj@SimpleTxRxSequence(varargin{:});
            
            nTx = length(obj.txAngle);
            
            % Manage the txFocus size
            if length(obj.txFocus)~=nTx
                if isscalar(obj.txFocus)
                    obj.txFocus = obj.txFocus .* ones(1,nTx);
                else
                    error("ARRUS:IllegalArgument", ...
                          "PWISequence: txFocus must be scalar or be the same size as txAngle");
                end
            end
            
            % Manage the txApertureCenter size
            if length(obj.txApertureCenter)~=nTx
                if isscalar(obj.txApertureCenter)
                    obj.txApertureCenter = obj.txApertureCenter .* ones(1,nTx);
                else
                    error("ARRUS:IllegalArgument", ...
                          "PWISequence: txApertureCenter must be scalar or be the same size as txAngle");
                end
            end
            
            % Fix the txFocus value
            if any(~isinf(obj.txFocus))
                obj.txFocus(:) = inf;
                warning("PWISequence: txFocus is forced to be all inf");
            end
            
        end
        
    end
end

