classdef STASequence < SimpleTxRxSequence
    % Synthetic Transmit Aperture Tx/Rx Sequence.
    %
    % :param rxCenterElement: vector of rx aperture center elements [element]. \
    %   When empty, rx aperture center will be set to the first element.
    % :param rxApertureCenter: vector of rx aperture center positions [m]. \
    %   When empty, rx aperture center will be set to the first (0) element.
    % :param rxApertureSize: size of the rx aperture [element]. \
    %   When 0 and rxCenterElement and rxApertureCenter are empty, all  \
    %   RX elements will be active.
    
    methods
        function obj = STASequence(varargin)
            obj = obj@SimpleTxRxSequence(varargin{:});
            
            % Default rx aperture
            if isempty(obj.rxCenterElement) && isempty(obj.rxApertureCenter) && obj.rxApertureSize == 0
                obj.rxCenterElement = [];
                obj.rxApertureCenter = 0.0 .* ones(1, length(obj.txAngle));
                obj.rxApertureSize = "nElements";
            end
            
        end
        
    end
end

