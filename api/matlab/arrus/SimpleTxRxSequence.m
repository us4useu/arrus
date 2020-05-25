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
    % :param rxDepthRange: defines the end (if scalar) or
    %                      the begining and the end (if two-element vector)
    %                      of the acquisition expressed by depth range [m]
    % :param rxNSamples: number of recorded samples per channel [sample]
    % :param nRepetitions: number of repetitions of the sequence (positive integer)
    % :param txPri: tx pulse repetition interval [s]
    % :param tgcStart: TGC starting gain [dB]
    % :param tgcSlope: TGC gain slope [dB/m]
    % 
    % TGC gain = tgcStart + tgcSlope * propagation distance
    % TGC gain is limited to 14-54 dB, any values out of that range
    % will be set to 54 dB (if > 54 dB) or 14 dB (if <14 dB)
    
    properties
        txCenterElement (1,:) {mustBeFinite, mustBeReal}
        txApertureCenter (1,:) {mustBeFinite, mustBeReal}
        txApertureSize (1,1) {mustBeFinite, mustBeInteger, mustBeNonnegative}
        txFocus (1,:) {mustBeNonNan, mustBeReal}
        txAngle (1,:) {mustBeFinite, mustBeReal}
        speedOfSound (1,1) {mustBeProperNumber}
        txFrequency (1,1) {mustBeProperNumber}
        txNPeriods (1,1) {mustBeInteger, mustBeProperNumber}
        rxDepthRange (1,:) 
        rxNSamples (1,:)
        nRepetitions (1,1) {mustBeInteger, mustBePositive} = 1
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
            
            checkProperties(obj)
        end
    end
end



function mustBeProperNumber(a)
    mustBeNonnegative(a)
    mustBeFinite(a)
%     mustBeNonempty(a)
    mustBeReal(a)

end



function mustBeProperDepthRange(a)
    
    if length(a(:))>2
        error(['Value assigned to rxDepthRange property ', ... 
               'should be a scalar or two-element vector'])
    end

    if length(a(:)) == 2
        if a(2) <= a(1)
            error(['The second element of rxDepthRange property ', ...
                   'should be bigger than the first element.'])
        end
    end
    mustBeProperNumber(a)
end

function checkProperties(obj)

%     disp([num2str(isprop(obj, 'rxDepthRange')), num2str(isempty(obj.rxDepthRange))])
%     disp([num2str(isprop(obj, 'rxNSamples')), num2str(isempty(obj.rxNSamples))])

    % checking if both properties are given (which is bad)
    if ~xor(isempty(obj.rxDepthRange), isempty(obj.rxNSamples))
        disp(num2str(obj.rxDepthRange))
        disp(num2str(obj.rxNSamples))
        error(['There can be only one of ',...
               'rxDepthRange and rxNSamples properties',...
               'in the sequence, not both.'])
    end
    
    % checking rxNSamples property
    if isempty(obj.rxDepthRange)
        mustBeProperNumber(obj.rxNSamples)
        mustBeInteger(obj.rxNSamples)
        mustBePositive(obj.rxNSamples)
    end
    
    % checking rxDepthRange property
    if isempty(obj.rxNSamples)
        mustBeProperDepthRange(obj.rxDepthRange)
    end


end