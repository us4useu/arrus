classdef SimpleTxRxSequence < Operation
    % A sequence of Tx/Rx operations to perform on a device.
    %
    
    properties
        txCenterElement
        txApertureCenter
        txApertureSize
        txFocus
        txAngle
        speedOfSound
        txFrequency
        txNPeriods
        rxNSamples (1,1) ...
                   {mustBeInteger,...
                   mustBePositive,...
                   mustBeDivisible(rxNSamples, 1024)} = 4*1024
        txPri (1,1) double {mustBePositive} = 100e-6
    end
    
    methods
        function obj = SimpleTxRxSequence(varargin)
            % assign property to the object
            if mod(nargin, 2) == 1
                throw( ...
                    MException( ...
                        "Arrus:params", ...
                        "Input should be a list of  'key', value params."))
            end
            for i = 1:2:nargin
                obj.(varargin{i}) = varargin{i+1};
            end
        end
    end
end

