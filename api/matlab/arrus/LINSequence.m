classdef LINSequence < SimpleTxRxSequence
   % Classical imaging Tx/Rx Sequence.
   %
   % :param rxCenterElement: vector of rx aperture center elements [element]. \
   %   When empty, will be the same as txCenterElement.
   % :param rxApertureCenter: vector of rx aperture center positions [m]. \
   %   When empty, will be the same as txApertureCenter.
   % :param rxApertureSize: size of the rx aperture [element]. \
   %   When empty, will be the same as txApertureSize.
    
    methods
        function obj = LINSequence(varargin)
            obj = obj@SimpleTxRxSequence(varargin{:});
            
            % Default rx aperture
            if isempty(obj.rxCenterElement) && isempty(obj.rxApertureCenter) && obj.rxApertureSize == 0
                obj.rxCenterElement = obj.txCenterElement;
                obj.rxApertureCenter = obj.txApertureCenter;
                obj.rxApertureSize = obj.txApertureSize;
                disp(['Using default Rx aperture of size ', num2str(obj.rxApertureSize)]);
            end
            
        end
        
    end
end

