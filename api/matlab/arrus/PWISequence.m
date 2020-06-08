classdef PWISequence < SimpleTxRxSequence
    % Plane Wave Imaging Tx/Rx Sequence.
    %
    % The PWI sequence has txFocus set to inf by default.
    % 
    % :param rxCenterElement: vector of rx aperture center elements [element]. \
    %   When empty, rx aperture center will be set to the first element.
    % :param rxApertureCenter: vector of rx aperture center positions [m]. \
    %   When empty, rx aperture center will be set to the first (0) element.
    % :param rxApertureSize: size of the rx aperture [element]. \
    %   When 0 and rxCenterElement and rxApertureCenter are empty all  \
    %   RX elements will be active.
    
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

