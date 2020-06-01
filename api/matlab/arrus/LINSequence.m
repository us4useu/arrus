classdef LINSequence < SimpleTxRxSequence
   % Classical imaging Tx/Rx Sequence.
    %
    % The length of the LIN sequence is determined by the length of txCenterElement.
    % The txFocus and txAngle must be scalars or be the same length as txCenterElement.
    
    methods
        function obj = LINSequence(varargin)
            obj = obj@SimpleTxRxSequence(varargin{:});
            
            nTx = length(obj.txCenterElement);
            
            % Manage the txFocus size
            if length(obj.txFocus)~=nTx
                if isscalar(obj.txFocus)
                    obj.txFocus = obj.txFocus .* ones(1,nTx);
                else
                    error("ARRUS:IllegalArgument", ...
                          "LINSequence: txFocus must be scalar or be the same size as txCenterElement");
                end
            end
            
            % Manage the txAngle size
            if length(obj.txAngle)~=nTx
                if isscalar(obj.txAngle)
                    obj.txAngle = obj.txAngle .* ones(1,nTx);
                else
                    error("ARRUS:IllegalArgument", ...
                          "LINSequence: txAngle must be scalar or be the same size as txCenterElement");
                end
            end
            
        end
        
    end
end

