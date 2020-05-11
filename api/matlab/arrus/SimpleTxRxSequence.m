classdef SimpleTxRxSequence < Operation
    % A sequence of Tx/Rx operations to perform on a device.
    %
    % :param txApertureCenter: vector of tx aperture center positions [m]
    % :param txApertureSize: size of the tx aperture [element]
    % :param txFocus: tx focal length [m]
    % :param txAngle: tx angle [rad]
    % :param speedOfSound: speed of sound for [m/s]
    % :param txFrequency: tx frequency [Hz]
    % :param txNPeriods: number of sine periods in the tx burst (can be 0.5, 1, 1.5, etc.)
    % :param rxNSamples: number of recorded samples per channel [sample]
    % :param txPri: tx pulse repetition interval [us] \
    %   **NOTE: in 0.4.0 (after release candidates stage) the value will be in [s]** \
    %   We will inform you when a change occurs.
    % :param tgcStart: TGC starting gain [dB]
    % :param tgcSlope: TGC gain slope [dB/m]
    % 
    % TGC gain = tgcStart + tgcSlope * propagation distance
    % TGC gain is limited to 14-54 dB, any values out of that range
    % will be set to 54 dB (if > 54 dB) or 14 dB (if <14 dB)
    
    properties
        txCenterElement (1,:)
        txApertureCenter (1,:)
        txApertureSize (1,1)
        txFocus (1,:)
        txAngle (1,:)
        speedOfSound (1,1)
        txFrequency (1,1)
        txNPeriods (1,1)
        rxNSamples (1,1) ...
                   {mustBeInteger,...
                   mustBePositive,...
                   mustBeDivisible(rxNSamples, 1024)} = 4*1024
        txPri (1,1) double {mustBePositive} = 100e-6
        tgcStart (1,1)
        tgcSlope (1,1)
    end
    
    methods
        function obj = SimpleTxRxSequence(varargin)
            if mod(nargin, 2) == 1
                error("Arrus:params", ...
                      "Input should be a list of  'key', value params.");
            end
            for i = 1:2:nargin
                obj.(varargin{i}) = varargin{i+1};
            end
        end
    end
end

