classdef CustomTxRxSequence
    % A class that stores parameters of the sequence of Tx/Rx \
    % operations to be performed on the device.
    
    properties
        txCenterElement (1,:) {mustBeFinite, mustBeReal}
        txApertureCenter (1,:) {mustBeFinite, mustBeReal}
        txApertureSize (1,:) {mustBeFinite, mustBeInteger, mustBeNonnegative}
        rxCenterElement (1,:) {mustBeFinite, mustBeReal}
        rxApertureCenter (1,:) {mustBeFinite, mustBeReal}
        rxApertureSize (1,1) {mustBeFinite, mustBeInteger, mustBeNonnegative}
        txFocus (1,:) {mustBeNonNan, mustBeReal}
        txAngle (1,:) {mustBeFinite, mustBeReal}
        speedOfSound (1,1) {mustBeProperNumber}
        txVoltage  (:,:) {mustBeNonnegative} = 0;
        txVoltageId (1,:) = 2
        txFrequency (1,:) = []
        txNPeriods (1,:) = []
        rxDepthRange (1,:) {mustBeProperNumber}
        rxNSamples (1,:) {mustBeFinite, mustBeInteger, mustBePositive}
        hwDdcEnable (1,1) {mustBeLogical} = true
        decimation (1,:) {mustBeFinite, mustBeInteger, mustBePositive}
        nRepetitions (1,1) {mustBeFinite, mustBeInteger, mustBePositive} = 1
        txPri (1,:) double {mustBePositive}
        tgcStart (1,:)
        tgcSlope (1,1) = 0
        txInvert (1,:) {mustBeLogical} = false
        workMode {mustBeTextScalar} = "MANUAL"
        sri (1,1) {mustBeNonnegative, mustBeFinite, mustBeReal} = 0
        bufferSize (1,1) {mustBeFinite, mustBeInteger, mustBePositive} = 2
        txWaveform = []
    end
    
    methods
        function obj = CustomTxRxSequence(varargin)
            % Creates a CustomTxRxSequence object.
            % 
            % Syntax:
            % obj = CustomTxRxSequence(name, value, ..., name, value)
            % 
            % All inputs are organized in name-value pairs.
            % 
            % :param txCenterElement: Center elements of the Tx apertures [elem]. \
            %   Numerical vector.
            % :param txApertureCenter: Center positions of the Tx apertures [m]. \
            %   Numerical vector.
            % :param txApertureSize: Sizes of the Tx apertures [elem]. \
            %   Numerical vector.
            % :param rxCenterElement: Center elements of the Rx apertures [elem]. \
            %   Numerical vector.
            % :param rxApertureCenter: Center positions of the Rx apertures [m]. \
            %   Numerical vector.
            % :param rxApertureSize: Size of the Rx apertures [elem]. \
            %   Numerical scalar.
            % :param txFocus: Tx focal distances [m]. Numerical vector.
            % :param txAngle: Tx angles [rad]. Numerical vector.
            % :param speedOfSound: Speed of sound determining the Tx delay \
            %   profiles [m/s]. Numerical scalar.
            % :param txVoltage: Tx voltage level [V]. Can be: \
            %   a scalar (pulse voltage range is [-txVoltage +txVoltage] for the \
            %   whole sequence), or a 2x2 array (defines two sets of negative and \
            %   positive Tx voltage amplitudes: [v1neg, v1pos; v2neg, v2pos]; the \
            %   voltage range can be selected individually for each Tx using txVoltageId). \
            %   txVoltage must always be nonnegative and v1 must be higher than v2. \
            %   "Legacy" systems only support scalar txVoltage.
            % :param txVoltageId: Tx voltage level identifiers (can be 1 \
            %   for range [-v1neg +v1pos], or 2 for [-v2neg +v2pos]). \
            %   Numerical vector.
            % :param txFrequency: Tx frequencies [Hz]. Numerical vector.
            % :param txNPeriods: Numbers of sine periods in the Tx burst \
            %   (can be 0.5, 1, 1.5, etc.). Numerical vector.
            % :param rxDepthRange: Acquisition depth range [m]. If scalar, \
            %   it defines the upper depth limit (the lower one is set to 0). \
            %   If 2-elem vector, it defines the lower and upper depth limits. \
            %   Numerical scalar/2-elem vector.
            % :param rxNSamples: Number of acquired samples. Numerical scalar.
            % :param hwDdcEnable: Enables hardware DDC (Digital Down Conversion). \
            %   It results in complex iq output data. Logical scalar.
            % :param decimation: Hardware decimation factor. Numerical \
            %   scalar, positive integer.
            % :param nRepetitions: Number of repetitions of the sequence. \
            %   Numerical scalar, positive integer.
            % :param txPri: Tx pulse repetition interval [s]. Numerical scalar.
            % :param tgcStart: TGC starting gain [dB]. Numerical scalar.
            % :param tgcSlope: TGC gain slope [dB/m]. Numerical scalar.
            % :param txInvert: Tx pulse polarity inversion. Logical vector.
            % :param workMode: System mode of operation. String scalar, \
            %   can be "MANUAL", "HOST", "SYNC", or "ASYNC".
            % :param sri: Sequence repeting interval [s]. Numerical scalar.
            % :param bufferSize: number of buffer elements (each element contains \
            %   data for a single sequence execution). Numerical scalar.
            % :param txWaveform: TX waveform to use. Scalar object of \
            %   arrus.ops.us4r.Waveform class.
            % 
            % TGC gain = tgcStart + tgcSlope * propagation distance

            if mod(nargin, 2) == 1
                error("ARRUS:params", ...
                      "Input should be a list of  'key', value params.");
            end
            for i = 1:2:nargin
                obj.(varargin{i}) = varargin{i+1};
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
            obj.workMode = upper(string(obj.workMode));
            if ~any(strcmp(obj.workMode,["MANUAL","HOST","SYNC","ASYNC"]))
                error("ARRUS:IllegalArgument", ...
                      "workMode must be one of the following: MANUAL, HOST, SYNC, or ASYNC.");
            end

            if ~xor(isempty(obj.txWaveform), isempty(obj.txFrequency) && isempty(obj.txNPeriods) && ~obj.txInvert)
                error("ARRUS:IllegalArgument", ...
                "Exactly one of the following should be provided: txWaveform or (txFrequency, txNPeriods, and txInvert)");
            end

            if ~isempty(obj.txWaveform) && obj.hwDdcEnable
                error("ARRUS:IllegalArgument", ...
                "hwDdcEnable must be set to false if txWaveform is provided");
            end

            %% Check size compatibility of aperture/focus/angle parameters
            nTx = max([	length(obj.txCenterElement) ...
                        length(obj.txApertureCenter) ...
                        length(obj.rxCenterElement) ...
                        length(obj.rxApertureCenter) ...
                        length(obj.txApertureSize) ...
                        length(obj.txFocus) ...
                        length(obj.txAngle)]);

            if isempty(obj.txWaveform)
                nTx = max([nTx length(obj.txFrequency) length(obj.txNPeriods) length(obj.txInvert)]);
            end
            
            obj.txCenterElement     = mustBeProperLength(obj.txCenterElement,nTx);
            obj.txApertureCenter    = mustBeProperLength(obj.txApertureCenter,nTx);
            obj.txApertureSize      = mustBeProperLength(obj.txApertureSize,nTx);
            obj.rxCenterElement     = mustBeProperLength(obj.rxCenterElement,nTx);
            obj.rxApertureCenter    = mustBeProperLength(obj.rxApertureCenter,nTx);
            obj.txFocus             = mustBeProperLength(obj.txFocus,nTx);
            obj.txAngle             = mustBeProperLength(obj.txAngle,nTx);
            obj.txVoltageId         = mustBeProperLength(obj.txVoltageId,nTx);

            %% txVoltage & txVoltageId validation
            mustBeProperNumber(obj.txVoltage);
            if ~ismatrix(obj.txVoltage) || (~isscalar(obj.txVoltage) && ~all(size(obj.txVoltage)==[2 2]))
                error("ARRUS:IllegalArgument", 'txVoltage must be scalar or 2x2 array');
            end
            if ~isscalar(obj.txVoltage) && any(obj.txVoltage(1,:) >= obj.txVoltage(2,:))
                error("ARRUS:IllegalArgument", 'txVoltage(2,:) must be higher than txVoltage(1,:)');
            end

            %% Pulse validation
            if isempty(obj.txWaveform)
                if isempty(obj.txInvert)
                    obj.txInvert = false;
                end

                obj.txFrequency         = mustBeProperLength(obj.txFrequency,nTx);
                obj.txNPeriods          = mustBeProperLength(obj.txNPeriods,nTx);
                obj.txInvert            = mustBeProperLength(obj.txInvert,nTx);
                obj.txInvert            = double(obj.txInvert);
            else
                obj.txInvert            = [];
            end
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

