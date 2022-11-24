classdef SimpleTxRxSequence < Operation
    % A sequence of Tx/Rx operations to perform on a device.
    %
    % :param txCenterElement: vector of tx aperture center elements [element]
    % :param txApertureCenter: vector of tx aperture center positions [m]
    % :param txApertureSize: vector of tx aperture sizes element]
    % :param rxCenterElement: vector of rx aperture center elements [element]
    % :param rxApertureCenter: vector of rx aperture center positions [m].
    % :param rxApertureSize: size of the rx aperture [element]
    % :param txFocus: vector of tx focal lengths [m]
    % :param txAngle: vector of tx angles [rad]
    % :param speedOfSound: speed of sound for [m/s]
    % :param txVoltage: tx voltage [V]
    % :param txFrequency: vector of tx frequencies [Hz]
    % :param txNPeriods: vector of numbers of sine periods in the tx burst (can be 0.5, 1, 1.5, etc.)
    % :param rxDepthRange: defines the end (if scalar) or
    %   the begining and the end (if two-element vector) \ 
    %   of the acquisition expressed by depth range [m]
    % :param rxNSamples: number of samples (if scalar) or 
    %   starting and ending sample numbers (if 2-element vector) \ 
    %   of recorded signal [sample]
    % :param nRepetitions: number of repetitions of the sequence (positive \
    %   integer). Can be a string "max" -- in this case, the maximum number \
    %   of repetitions that can be performed on the system (taking into \ 
    %   account the size of RAM, etc.) will be used.
    % :param txPri: tx pulse repetition interval [s]
    % :param tgcStart: TGC starting gain [dB]
    % :param tgcSlope: TGC gain slope [dB/m]
    % :param fsDivider: sampling frequency divider. Should be positive int. \ 
    %   Default value is equal to 1, which means sampling with \
    %   the highest possible frequency (no decimation)
    % :param txInvert: tx pulse polarity marker
    % 
    % TGC gain = tgcStart + tgcSlope * propagation distance
    % TGC gain is limited to 14-54 dB, any values out of that range
    % will be set to 54 dB (if > 54 dB) or 14 dB (if <14 dB)
    
    properties
        txCenterElement (1,:) {mustBeFinite, mustBeReal}
        txApertureCenter (1,:) {mustBeFinite, mustBeReal}
        txApertureSize (1,:)
        rxCenterElement (1,:) {mustBeFinite, mustBeReal}
        rxApertureCenter (1,:) {mustBeFinite, mustBeReal}
        rxApertureSize (1,1)
        txFocus (1,:) {mustBeNonNan, mustBeReal}
        txAngle (1,:) {mustBeFinite, mustBeReal}
        speedOfSound (1,1) {mustBeProperNumber}
        txVoltage (1,1) {mustBeNonnegative} = 0;
        txFrequency (1,:) {mustBeProperNumber}
        txNPeriods (1,:) {mustBeProperNumber}
        rxDepthRange (1,:) {mustBeProperNumber}
        rxNSamples (1,:) {mustBeFinite, mustBeInteger, mustBePositive}
        nRepetitions (1,:) = 1
        txPri (1,:) double {mustBePositive}
        tgcStart (1,:)
        tgcSlope (1,:)
        fsDivider(1,1) {mustBeInteger, mustBePositive, ...
            mustBeLessThan(fsDivider, 257)} = 1
        txInvert (1,:) {mustBeLogical} = false
    end
    
    methods
        function obj = SimpleTxRxSequence(varargin)
            if mod(nargin, 2) == 1
                error("ARRUS:params", ...
                      "Input should be a list of  'key', value params.");
            end
            for i = 1:2:nargin
                obj.(varargin{i}) = varargin{i+1};
            end
            
            if ischar(obj.nRepetitions)
                obj.nRepetitions = convertCharsToStrings(obj.nRepetitions);
            end
            
            % Validate.
            mustBeXor(obj,{'txCenterElement','txApertureCenter'});
            if ~isempty(obj.rxCenterElement) || ~isempty(obj.rxApertureCenter)
                mustBeXor(obj,{'rxCenterElement','rxApertureCenter'});
            end
            mustBeXor(obj,{'rxDepthRange','rxNSamples'});
            obj.rxDepthRange = mustBeLimit(obj,'rxDepthRange',0);
            obj.rxNSamples = mustBeLimit(obj,'rxNSamples',1);
            
            % Specific validations
            mustBeIntOrStr(obj,'nRepetitions',1,"max");
            mustBeIntOrStr(obj,'txApertureSize',0,"nElements");
            mustBeIntOrStr(obj,'rxApertureSize',0,["nChannels","nElements"]);
            
            %% Check size compatibility of aperture/focus/angle parameters
            nTx = max([	length(obj.txCenterElement) ...
                        length(obj.txApertureCenter) ...
                        length(obj.rxCenterElement) ...
                        length(obj.rxApertureCenter) ...
                        length(obj.txApertureSize) ...
                        length(obj.txFocus) ...
                        length(obj.txAngle) ...
                        length(obj.txFrequency) ...
                        length(obj.txNPeriods) ...
                        length(obj.txInvert) ]);
            
            obj.txCenterElement     = mustBeProperLength(obj.txCenterElement,nTx);
            obj.txApertureCenter	= mustBeProperLength(obj.txApertureCenter,nTx);
            obj.rxCenterElement     = mustBeProperLength(obj.rxCenterElement,nTx);
            obj.rxApertureCenter	= mustBeProperLength(obj.rxApertureCenter,nTx);
            obj.txFocus             = mustBeProperLength(obj.txFocus,nTx);
            obj.txAngle             = mustBeProperLength(obj.txAngle,nTx);
            obj.txFrequency         = mustBeProperLength(obj.txFrequency,nTx);
            obj.txNPeriods          = mustBeProperLength(obj.txNPeriods,nTx);
            obj.txInvert            = mustBeProperLength(obj.txInvert,nTx);
            if ~isstring(obj.txApertureSize)
                obj.txApertureSize	= mustBeProperLength(obj.txApertureSize,nTx);
            end
            
            obj.txInvert = double(obj.txInvert);
            
        end
    end
end


function mustBeProperNumber(a)
    mustBeNonnegative(a)
    mustBeFinite(a)
    mustBeReal(a)
end

function mustBeLogical(a)
    if ~islogical(a) && ~all(any(a==[0;1],1),2)
        error('txInvert property must be equal one of the following: true, false, 1 or 0.')
    end
        
end

function mustBeXor(obj,fieldNames)
    
    nField = length(fieldNames);
    
    fieldIsNonEmpty = false(1,nField);
    for iField=1:nField
        fieldIsNonEmpty(iField) = ~isempty(obj.(fieldNames{iField}));
    end
    
    if sum(fieldIsNonEmpty) ~= 1
        error("ARRUS:params", ...
            ['One and only one of {' char(join(fieldNames,', ')) '} must be defined.'])
    end
end

function mustBeIntOrStr(obj,fieldName,intDomain,strDomain)
    % intDomain: 0-nonnegative integers; 1-positive integers
    % strDomain: array of accepted string values
    
    argIn = obj.(fieldName);
    
    if isnumeric(argIn)
        mustBeInteger(argIn);
        if intDomain
            mustBePositive(argIn);
        else
            mustBeNonnegative(argIn);
        end
    elseif isstring(argIn)
        mustBeMember(argIn,strDomain);
    else
        error(['Unhandled data type of ' fieldName ': ', ...
            class(argIn)])
    end
    
end

function argOut = mustBeProperLength(argIn,propLength)
    
    if isempty(argIn) || length(argIn)==propLength
        argOut = argIn;
    elseif isscalar(argIn)
        argOut = argIn .* ones(1,propLength);
    else
        error("ARRUS:IllegalArgument", ...
                          "Incompatible parameter length");
    end
    
end

function argOut = mustBeLimit(obj,fieldName,defLo)
    
    argIn = obj.(fieldName);
    
    if isempty(argIn)
        argOut = argIn;
    else
        % check the size
        if numel(argIn)>2
            error("Arrus:params", ...
                ['Value assigned to ' fieldName ' property ', ...
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
                ['The ' fieldName ' property should have ascending order']);
        end
    end

end

