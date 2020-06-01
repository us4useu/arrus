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
    % :param rxNSamples: number of samples (if scalar) or 
    %                      starting and ending sample numbers (if 2-element vector) 
    %                      of recorded signal [sample]
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
        rxDepthRange (1,:) {mustBeProperNumber}
        rxNSamples (1,:) {mustBeFinite, mustBeInteger, mustBePositive}
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
            
            mustBeXor(obj.rxDepthRange,obj.rxNSamples);
            obj.rxDepthRange = mustBeLimit(obj.rxDepthRange,0);
            obj.rxNSamples = mustBeLimit(obj.rxNSamples,1);
            
        end
    end
end


function mustBeProperNumber(a)
    mustBeNonnegative(a)
    mustBeFinite(a)
    mustBeReal(a)
end

function mustBeXor(varargin)
    
    % input argument names
    argNames = strings(1,nargin);
    for iArg=1:nargin
        argNames(iArg) = inputname(iArg);
        if contains(argNames(iArg),'.')
            argNames(iArg) = extractAfter(argNames(iArg),'.');
        end
    end
    
    % error if more/less than one argument is non-empty
    argIsNonEmpty = ~cellfun(@isempty,varargin);
    if sum(argIsNonEmpty) ~= 1
        error("Arrus:params", ...
            ['One and only one of: ' join(argNames,', ') ' must be defined.'])
    end
end

function argOut = mustBeLimit(argIn,defLo)
    
    argName = inputname(1);
    if contains(argName,'.')
        argName = extractAfter(argName,'.');
    end
    
    if isempty(argIn)
        argOut = argIn;
    else
        % check the size
        if length(argIn(:))>2
            error("Arrus:params", ...
                ['Value assigned to ' argName ' property ', ...
                'should be a scalar or two-element vector'])
        end
        
        % expand scalar to 2-element vector
        if isscalar(argIn)
            argOut = [defLo argIn];
        else
            argOut = argIn;
        end
        
        % check the ascending order
        if argOut(2) <= argOut(1)
            error("Arrus:params", ...
                ['The ' argName ' property should have ascending order']);
        end
    end

end

