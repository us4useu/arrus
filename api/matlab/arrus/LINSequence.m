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
            
        end
        
    end
end

