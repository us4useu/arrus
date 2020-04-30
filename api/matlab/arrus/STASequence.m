classdef STASequence < SimpleTxRxSequence
    % Synthetic Transmit Aperture Tx/Rx Sequence.
    %
    % The length of the STA sequence is determined by the length of txApertureCenter.
    % The txFocus and txAngle must be scalars or be the same length as txApertureCenter.
    
    methods
        function obj = STASequence(varargin)
            obj = obj@SimpleTxRxSequence(varargin{:});
            
            nTx = length(obj.txApertureCenter);
            
            % Manage the txFocus size
            if length(obj.txFocus)~=nTx
                if isscalar(obj.txFocus)
                    obj.txFocus = obj.txFocus .* ones(1,nTx);
                else
                    error("ARRUS:IllegalArgument", ...
                          "STASequence: txFocus must be scalar or be the same size as txApertureCenter");
                end
            end
            
            % Manage the txAngle size
            if length(obj.txAngle)~=nTx
                if isscalar(obj.txAngle)
                    obj.txAngle = obj.txAngle .* ones(1,nTx);
                else
                    error("ARRUS:IllegalArgument", ...
                          "STASequence: txAngle must be scalar or be the same size as txApertureCenter");
                end
            end
            
        end
        
    end
end

