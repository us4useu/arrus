classdef Us4R < handle
    % A handle to the Us4R system. 
    %
    % This class provides functions to configure the system and perform
    % data acquisition using the Us4R.
    % 
    % :param configFile: name of the prototxt file containing setup information.
    % :param voltage: a voltage to set, should be in range 0-90 [0.5*Vpp]. Optional. Will be replaced with txVoltage parameter in CustomTxRxSequence.
    % :param logTime: set to true if you want to display acquisition and reconstruction time. Optional.

    properties(Access = private)
        sys
        seq
        rec
        us4r
        session
        buffer
        logTime
    end
    
    methods

        function obj = Us4R(varargin)
            
            % Input parser
            paramsParser = inputParser;
            addParameter(paramsParser, 'configFile', [], @(x) validateattributes(x, {'char','string'}, {'scalartext'}, 'Us4R', 'configFile'));
            addParameter(paramsParser, 'interfEnable', false, @(x) validateattributes(x, {'logical'}, {'scalar'}, 'Us4R', 'interfEnable'));
            addParameter(paramsParser, 'voltage', 0, @(x) validateattributes(x, {'numeric'}, {'scalar','nonnegative'}, 'Us4R', 'voltage'));
            addParameter(paramsParser, 'logTime', false, @(x) validateattributes(x, {'logical'}, {'scalar'}, 'Us4R', 'logTime'));
            parse(paramsParser, varargin{:});
            
            configFile   = paramsParser.Results.configFile;
            interfEnable = paramsParser.Results.interfEnable;
            voltage      = paramsParser.Results.voltage;
            logTime      = paramsParser.Results.logTime;
            
            if isempty(configFile) || ~isfile(configFile)
                [fileName,pathName,filterIndex] = uigetfile('*.prototxt','Select prototxt config file');
                if filterIndex	== 0
                    obj = [];
                    return;
                else
                    configFile = [pathName fileName];
                end
            end
            
            % Initialization
            arrus.initialize("clogLevel", "INFO", "logFilePath", "C:/Temp/arrus.log", "logFileLevel", "TRACE");
            
            obj.session = arrus.session.Session(configFile);
            obj.us4r = obj.session.getDevice("/Us4R:0");
            
            obj.sys.nChArius = 32;
            obj.sys.rxSampFreq = 65e6;
            obj.sys.voltage = voltage;
            obj.logTime = logTime;
            
            % Probe parameters
            probe = obj.us4r.getProbeModel;
            obj.sys.nElem = double(probe.nElements);
            obj.sys.pitch = probe.pitch;
            obj.sys.freqRange = double(probe.txFrequencyRange);
            obj.sys.maxVpp = double(probe.voltageRange(2));
            obj.sys.curvRadius = probe.curvatureRadius;
            
            % Validate voltage
            if 2*obj.sys.voltage > obj.sys.maxVpp
                error(['Voltage exceeds the safe limit. ', ...
                       'For the current probe the limit is ', ...
                       num2str(obj.sys.maxVpp/2), '[V].' ...
                       ])
            end

             % Position (pos,x,z) and orientation (ang) of each probe element
             obj.sys.posElem = (-(obj.sys.nElem-1)/2 : (obj.sys.nElem-1)/2) * obj.sys.pitch; % [m] (1 x nElem) position of probe elements along the probes surface
             if obj.sys.curvRadius == 0
                 obj.sys.angElem = zeros(1,obj.sys.nElem); % [rad] (1 x nElem) orientation of probe elements
                 obj.sys.xElem = obj.sys.posElem; % [m] (1 x nElem) z-position of probe elements
                 obj.sys.zElem = zeros(1,obj.sys.nElem);% [m] (1 x nElem) x-position of probe elements
             else
                 obj.sys.angElem = obj.sys.posElem / -obj.sys.curvRadius;
                 obj.sys.xElem = -obj.sys.curvRadius * sin(obj.sys.angElem);
                 obj.sys.zElem = -obj.sys.curvRadius * cos(obj.sys.angElem);
                 obj.sys.zElem = obj.sys.zElem - min(obj.sys.zElem);
             end
             
             obj.sys.interfEnable = interfEnable;
             if obj.sys.interfEnable
                 wedge = wedgeParams();
                 obj.sys.interfSize = wedge.interfSize;
                 obj.sys.interfAng  = wedge.interfAng;
                 obj.sys.interfSos  = wedge.interfSos;
                 
                 obj.sys.angElem = obj.sys.angElem + obj.sys.interfAng;
                 
                 xElemNoInterf = obj.sys.xElem;
                 zElemNoInterf = obj.sys.zElem;
                 obj.sys.xElem = xElemNoInterf * cos(obj.sys.interfAng) ...
                               + zElemNoInterf * sin(obj.sys.interfAng);
                 obj.sys.zElem = zElemNoInterf * cos(obj.sys.interfAng) ...
                               - xElemNoInterf * sin(obj.sys.interfAng) ...
                               - obj.sys.interfSize;
             end
             
             obj.sys.tangElem = tan(obj.sys.angElem);
            
             obj.sys.isHardwareProgrammed = false;
        end
        
        function closeSession(obj)
            obj.session.close();
        end
        
        function nProbeElem = getNProbeElem(obj)
            nProbeElem = obj.sys.nElem;
        end

        function upload(obj, sequenceOperation, reconstructOperation, enableHardwareProgramming)
            % Uploads operations to the us4R system.
            %
            % Supports :class:`CustomTxRxSequence`
            % and :class:`Reconstruction` implementations.
            %
            % :param sequenceOperation: TX/RX sequence to perform on the us4R system
            % :param reconstructOperation: reconstruction to perform with the collected data
            % :param enableHardwareProgramming: determines if the hardware
            % is programmed or not (optional, default = true)
            % :returns: updated Us4R object
            
            if ~isa(sequenceOperation,'CustomTxRxSequence')
                error("ARRUS:IllegalArgument", ...
                      'Invalid sequence object, must be CustomTxRxSequence');
            end

            if nargin>=3 && ~isempty(reconstructOperation) && ~isa(reconstructOperation,'Reconstruction')
                error("ARRUS:IllegalArgument", ...
                      'Invalid reconstruction object, must be Reconstruction');
            end
            
            obj.setSeqParams(...
                'txCenterElement', sequenceOperation.txCenterElement, ...
                'txApertureCenter', sequenceOperation.txApertureCenter, ...
                'txApertureSize', sequenceOperation.txApertureSize, ...
                'rxCenterElement', sequenceOperation.rxCenterElement, ...
                'rxApertureCenter', sequenceOperation.rxApertureCenter, ...
                'rxApertureSize', sequenceOperation.rxApertureSize, ...
                'txFocus', sequenceOperation.txFocus, ...
                'txAngle', sequenceOperation.txAngle, ...
                'speedOfSound', sequenceOperation.speedOfSound, ...
                'txVoltage', sequenceOperation.txVoltage, ...
                'txFrequency', sequenceOperation.txFrequency, ...
                'txNPeriods', sequenceOperation.txNPeriods, ...
                'rxDepthRange', sequenceOperation.rxDepthRange, ...
                'rxNSamples', sequenceOperation.rxNSamples, ...
                'hwDdcEnable', sequenceOperation.hwDdcEnable, ...
                'decimation', sequenceOperation.decimation, ...
                'nRepetitions', sequenceOperation.nRepetitions, ...
                'txPri', sequenceOperation.txPri, ...
                'tgcStart', sequenceOperation.tgcStart, ...
                'tgcSlope', sequenceOperation.tgcSlope, ...
                'txInvert', sequenceOperation.txInvert);
            
            % Program hardware
            if nargin<4 || enableHardwareProgramming
                obj.programHW;
                obj.sys.isHardwareProgrammed = true;
            else
                error('Support for enableHardwareProgramming=false is temporarily suspended');
            end
            
            if nargin<3 || isempty(reconstructOperation)
                obj.rec.enable = false;
                return;
            end
                
            obj.setRecParams(...
                'gridModeEnable', reconstructOperation.gridModeEnable, ...
                'filterEnable', reconstructOperation.filterEnable, ...
                'filterACoeff', reconstructOperation.filterACoeff, ...
                'filterBCoeff', reconstructOperation.filterBCoeff, ...
                'filterDelay', reconstructOperation.filterDelay, ...
                'swDdcEnable', reconstructOperation.swDdcEnable, ...
                'decimation', reconstructOperation.decimation, ...
                'xGrid', reconstructOperation.xGrid, ...
                'zGrid', reconstructOperation.zGrid, ...
                'sos', reconstructOperation.sos, ...
                'rxApod', reconstructOperation.rxApod, ...
                'bmodeEnable', reconstructOperation.bmodeEnable, ...
                'colorEnable', reconstructOperation.colorEnable, ...
                'vectorEnable', reconstructOperation.vectorEnable, ...
                'bmodeFrames', reconstructOperation.bmodeFrames, ...
                'colorFrames', reconstructOperation.colorFrames, ...
                'vector0Frames', reconstructOperation.vector0Frames, ...
                'vector1Frames', reconstructOperation.vector1Frames, ...
                'bmodeRxTangLim', reconstructOperation.bmodeRxTangLim, ...
                'colorRxTangLim', reconstructOperation.colorRxTangLim, ...
                'vector0RxTangLim', reconstructOperation.vector0RxTangLim, ...
                'vector1RxTangLim', reconstructOperation.vector1RxTangLim, ...
                'wcFilterACoeff', reconstructOperation.wcFilterACoeff, ...
                'wcFilterBCoeff', reconstructOperation.wcFilterBCoeff, ...
                'wcFiltInitSize', reconstructOperation.wcFiltInitSize);
            
            obj.rec.enable = true;
            
        end
        
        function [sys, seq] = getImagingMetadata(obj)
            sys = obj.sys;
            seq = obj.seq;
        end
        
        function [rf,img] = run(obj)
            % Runs uploaded operations in the us4R system.
            %
            % Supports :class:`CustomTxRxSequence` and :class:`Reconstruction`
            % implementations.
            %
            % :returns: RF frame and reconstructed image (if :class:`Reconstruction` operation was uploaded)
            
            [rf, ~] = obj.execSequence;
            
            if obj.rec.enable
                img = obj.execReconstr(rf(:,:,:,1));
            else
                img = [];
            end
        end
        
        function [rf, img, metadata] = runWithMetadata(obj)
            % Runs uploaded operations in the us4R system.
            %
            % Supports :class:`CustomTxRxSequence` and :class:`Reconstruction`
            % implementations.
            %
            % :returns: RF frame, reconstructed image (if :class:`Reconstruction` operation was uploaded) and metadata located in the first sample of the master module
            [rf, metadata] = obj.execSequence;
            
            if obj.rec.enable
                img = obj.execReconstr(rf(:,:,:,1));
            else
                img = [];
            end
        end
        
        function runLoop(obj, isContinue, callback)
            % Runs the uploaded operations in a loop.
            % 
            % Supports :class:`CustomTxRxSequence` and \
            % :class:`Reconstruction` implementations.
            %
            % :param isContinue: should the system continue executing \
            %   the op? Takes no parameters and returns a boolean value.
            % :param callback: a function to call after executing the \
            %   operation. Should take one parameter, which will be feed with \
            %   the output of the executed op.
            
            i = 0;
            while(isContinue())
                i = i + 1;
                
                tic;
                rf = obj.execSequence;
                acqTime = toc;
                
                if obj.rec.enable
                    tic;
                    img = obj.execReconstr(rf(:,:,:,1));
                    recTime = toc;
                    callback(img);
                else
                    callback(rf);
                end
                
                if obj.logTime
                    disp(['Frame no. ' num2str(i)]);
                    disp(['Acq.  time = ' num2str(acqTime,         '%5.3f') ' s']);
                    if exist('recTime', 'var')
                        disp(['Rec.  time = ' num2str(recTime,         '%5.3f') ' s']);
                        disp(['Frame rate = ' num2str(1/(acqTime+recTime),'%5.1f') ' fps']);
                    end    
                    disp('--------------------');
                end
            end
        end
        
        function [img] = reconstructOffline(obj,rfRaw)
            img = obj.execReconstr(rfRaw);
        end
    end
    
    methods(Access = private)

        function setSeqParams(obj,varargin)

            %% Set sequence parameters
            % Sequence parameters names mapping
            %                    public name         private name
            seqParamMapping = { 'txCenterElement',  'txCentElem'; ...
                                'txApertureCenter', 'txApCent'; ...
                                'txApertureSize',   'txApSize'; ...
                                'rxCenterElement',  'rxCentElem'; ...
                                'rxApertureCenter', 'rxApCent'; ...
                                'rxApertureSize',   'rxApSize'; ...
                                'txFocus',          'txFoc'; ...
                                'txAngle',          'txAng'; ...
                                'speedOfSound',     'c'; ...
                                'txVoltage',        'txVoltage'; ...
                                'txFrequency',      'txFreq'; ...
                                'txNPeriods',       'txNPer'; ...
                                'rxDepthRange',     'dRange'; ...
                                'rxNSamples',       'nSamp'; ...
                                'hwDdcEnable',      'hwDdcEnable'; ...
                                'decimation',       'dec'; ...
                                'nRepetitions',     'nRep'; ...
                                'txPri',            'txPri'; ...
                                'tgcStart',         'tgcStart'; ...
                                'tgcSlope',         'tgcSlope'; ...
                                'txInvert'          'txInvert'};

            for iPar=1:size(seqParamMapping,1)
                obj.seq.(seqParamMapping{iPar,2}) = [];
            end

            nPar = length(varargin)/2;
            for iPar=1:nPar
                idPar = strcmpi(varargin{iPar*2-1},seqParamMapping(:,1));
                obj.seq.(seqParamMapping{idPar,2}) = reshape(varargin{iPar*2},1,[]);
            end
            
            %% Default decimation & DDC filter coefficients
            if obj.seq.hwDdcEnable
                if isempty(obj.seq.dec)
                    obj.seq.dec = round(obj.sys.rxSampFreq / max(obj.seq.txFreq));
                end
                obj.seq.fpgaDec = 1;
                
                cutoffFrequency = mean(obj.seq.txFreq)/(obj.sys.rxSampFreq/2);
                firOrder = obj.seq.dec * 16 - 1;
                firCoeff = fir1(firOrder, cutoffFrequency, "low");
                obj.seq.ddcFirCoeff = firCoeff((numel(firCoeff)/2 + 1) : end);
                
            else
                if isempty(obj.seq.dec)
                    obj.seq.dec = 1;
                end
                obj.seq.fpgaDec = obj.seq.dec;
                
            end
            
            %% Sampling frequency
            obj.seq.rxSampFreq	= obj.sys.rxSampFreq / obj.seq.dec; % [Hz] sampling frequency
            
            %% Number of Tx
            obj.seq.nTx	= length(obj.seq.txAng);
            
            %% rxNSamples & rxDepthRange
            % rxDepthRange was given in sequence (rxNSamples is empty)
            if isempty(obj.seq.nSamp)
                % convert from [m] to samples
                sampRange  = round(...
                    2*obj.seq.rxSampFreq*obj.seq.dRange/obj.seq.c ...
                    ) + 1;
                
                % rxNSamples (nSamp) must be coherent with rxDepthRange
                nSamp = sampRange(2) - sampRange(1) + 1;
                
                % nSamp must be dividible by 64 (for now)
                nSamp = 64*ceil(nSamp/64);
                obj.seq.nSamp = nSamp;
                
                obj.seq.startSample = sampRange(1);
            else
                obj.seq.startSample = obj.seq.nSamp(1);
                obj.seq.nSamp = diff(obj.seq.nSamp) + 1;
            end
            
            %% txPri
            if isempty(obj.seq.txPri)
                obj.seq.txPri = (obj.seq.startSample + obj.seq.nSamp) / obj.seq.rxSampFreq + 42e-6;
            end
            
            %% txVoltage
            if 2*obj.seq.txVoltage > obj.sys.maxVpp
                error(['txVoltage exceeds the safe limit. ', ...
                       'For the current probe the limit is ', ...
                       num2str(obj.sys.maxVpp/2), '[V].']);
            end
            obj.seq.txVoltage = max(obj.seq.txVoltage, obj.sys.voltage);
            
            %% TGC
            % Default TGC
            if isempty(obj.seq.tgcStart)
                obj.seq.tgcStart = 14;
            end
            if isempty(obj.seq.tgcSlope)
                obj.seq.tgcSlope = 2 * 0.5e2 * mean(obj.seq.txFreq)*1e-6;
            end
            
            distance = (400 : 150 : (obj.seq.startSample + obj.seq.nSamp - 1)*obj.seq.dec) / obj.sys.rxSampFreq * obj.seq.c;         % [m]
            obj.seq.tgcCurve = obj.seq.tgcStart + obj.seq.tgcSlope * distance;  % [dB]
            if any(obj.seq.tgcCurve < 14 | obj.seq.tgcCurve > 54)
                warning('TGC values are limited to 14-54dB range');
                obj.seq.tgcCurve = max(14,min(54,obj.seq.tgcCurve));
            end
            
            %% Tx/Rx aperture string/missing parameters
            if isstring(obj.seq.txApSize) && obj.seq.txApSize == "nElements"
                obj.seq.txApSize = obj.sys.nElem * ones(1,obj.seq.nTx);
                disp(['txApertureSize set to ' num2str(obj.seq.txApSize(1)) '.']);
            end
            
            if isstring(obj.seq.rxApSize) && obj.seq.rxApSize == "nElements"
                obj.seq.rxApSize = obj.sys.nElem;
                disp(['rxApertureSize set to ' num2str(obj.seq.rxApSize) '.']);
            end
            
            % delete: txApCent & rxApCent
            if isempty(obj.seq.txApCent)
                obj.seq.txApCent	= interp1(1:obj.sys.nElem, obj.sys.posElem, obj.seq.txCentElem);
            else
                obj.seq.txCentElem	= interp1(obj.sys.posElem, 1:obj.sys.nElem, obj.seq.txApCent);
            end
            obj.seq.txApCentZ	= interp1(1:obj.sys.nElem, obj.sys.zElem,   obj.seq.txCentElem);
            obj.seq.txApCentX	= interp1(1:obj.sys.nElem, obj.sys.xElem,   obj.seq.txCentElem);
            obj.seq.txApCentAng	= interp1(1:obj.sys.nElem, obj.sys.angElem, obj.seq.txCentElem);
            obj.seq.txAngZX     = obj.seq.txApCentAng + obj.seq.txAng;
            
            if isempty(obj.seq.rxApCent)
                obj.seq.rxApCent	= interp1(1:obj.sys.nElem, obj.sys.posElem, obj.seq.rxCentElem);
            else
                obj.seq.rxCentElem	= interp1(obj.sys.posElem, 1:obj.sys.nElem, obj.seq.rxApCent);
            end
            
            %% Validate sequence if wedge interface is used
            if obj.sys.interfEnable && any(obj.seq.txApSize~=1)
                error("setSeqParams: only SSTA scheme is supported when wedge interface is used");
            end
            
            if obj.sys.interfEnable && (numel(unique(obj.seq.txFreq)) > 1 || numel(unique(obj.seq.txNPer)) > 1)
                error("setSeqParams: txFrequency and txNPeriods must be constant when wedge interface is used");
            end
            
            %% Aperture masks & delays
            obj.calcTxRxApMask;
            obj.calcTxDelays;
            
            obj.seq.nSampOmit = (max(obj.seq.txDel) + obj.seq.txNPer./obj.seq.txFreq) * obj.seq.rxSampFreq + ceil(50 / obj.seq.dec);
            obj.seq.initDel   = - obj.seq.startSample/obj.seq.rxSampFreq + obj.seq.txDelCent + obj.seq.txNPer./(2*obj.seq.txFreq);
            if obj.seq.hwDdcEnable
                obj.seq.initDel   = obj.seq.initDel + (8+1)/obj.seq.rxSampFreq;
            end

        end

        function setRecParams(obj,varargin)
            %% Set reconstruction parameters
            % Reconstruction parameters names mapping
            %                    public name         private name
            recParamMapping = { 'gridModeEnable',   'gridModeEnable'; ...
                                'filterEnable',     'filtEnable'; ...
                                'filterACoeff',     'filtA'; ...
                                'filterBCoeff',     'filtB'; ...
                                'filterDelay',      'filtDel'; ...
                                'swDdcEnable',      'swDdcEnable'; ...
                                'decimation',       'dec'; ...
                                'xGrid',            'xGrid'; ...
                                'zGrid',            'zGrid'; ...
                                'sos',              'sos'; ...
                                'rxApod',           'rxApod'; ...
                                'bmodeEnable',      'bmodeEnable'; ...
                                'colorEnable',      'colorEnable'; ...
                                'vectorEnable',     'vectorEnable'; ...
                                'bmodeFrames',      'bmodeFrames'; ...
                                'colorFrames',      'colorFrames'; ...
                                'vector0Frames',	'vect0Frames'; ...
                                'vector1Frames',	'vect1Frames'; ...
                                'bmodeRxTangLim',	'bmodeRxTangLim'; ...
                                'colorRxTangLim',	'colorRxTangLim'; ...
                                'vector0RxTangLim',	'vect0RxTangLim'; ...
                                'vector1RxTangLim',	'vect1RxTangLim'; ...
                                'wcFilterACoeff',   'wcFiltA'; ...
                                'wcFilterBCoeff',   'wcFiltB'; ...
                                'wcFiltInitSize',   'wcFiltInitSize'};

            for iPar=1:size(recParamMapping,1)
                obj.rec.(recParamMapping{iPar,2}) = [];
            end

            nPar = length(varargin)/2;
            for iPar=1:nPar
                idPar = strcmpi(varargin{iPar*2-1},recParamMapping(:,1));
                obj.rec.(recParamMapping{idPar,2}) = reshape(varargin{iPar*2},1,[]);
            end
            
            %% Software DDC parameters
            if isempty(obj.rec.swDdcEnable)
                obj.rec.swDdcEnable = ~obj.seq.hwDdcEnable;
            end
            if obj.rec.swDdcEnable
                if obj.seq.hwDdcEnable
                    error("setRecParams: hwDdcEnable & swDdcEnable cannot be set to true at a time");
                end
                if isempty(obj.rec.dec)
                    obj.rec.dec = round(obj.seq.rxSampFreq / max(obj.seq.txFreq));
                end
                
                % Filter design the same as in hardware DDC
                % downConvertion.m performs filtration with no phase delay
                cutoffFrequency = mean(obj.seq.txFreq)/(obj.seq.rxSampFreq/2);
                firOrder = obj.rec.dec * 16 - 1;
                obj.rec.ddcFirCoeff = fir1(firOrder, cutoffFrequency, "low");
            else
                obj.rec.dec = 1;
            end
            
            %% Validate reconstruction if wedge interface is used
            if obj.sys.interfEnable && ~obj.rec.gridModeEnable
                error("setRecParams: only grid reconstruction is supported when wedge interface is used");
            end
            
            %% Default sos
            if isempty(obj.rec.sos)
                obj.rec.sos = obj.seq.c;
            end
            
            %% Validate frames selection
            if obj.rec.bmodeEnable && any(obj.rec.bmodeFrames > obj.seq.nTx)
                error("setRecParams: bmodeFrames refers to nonexistent transmission id");
            end
            
            if obj.rec.colorEnable && any(obj.rec.colorFrames > obj.seq.nTx)
                error("setRecParams: colorFrames refers to nonexistent transmission id");
            end
            
            if obj.rec.vectorEnable && any(obj.rec.vect0Frames > obj.seq.nTx)
                error("setRecParams: vector0Frames refers to nonexistent transmission id");
            end
            
            if obj.rec.vectorEnable && any(obj.rec.vect1Frames > obj.seq.nTx)
                error("setRecParams: vector1Frames refers to nonexistent transmission id");
            end
            
            %% Default bmodeFrames
            if obj.rec.bmodeEnable && isempty(obj.rec.bmodeFrames)
                obj.rec.bmodeFrames = 1:obj.seq.nTx;
            end
            
            %% Validate/adjust size of the RxTangLims
            obj.rec.bmodeRxTangLim = reshape(obj.rec.bmodeRxTangLim,[],2);
            if obj.rec.bmodeEnable
                if size(obj.rec.bmodeRxTangLim,1) == 1
                    obj.rec.bmodeRxTangLim = obj.rec.bmodeRxTangLim.*ones(numel(obj.rec.bmodeFrames),1);
                elseif size(obj.rec.bmodeRxTangLim,1) ~= numel(obj.rec.bmodeFrames)
                    error("setRecParams: number of rows in bmodeRxTangLim must equal the length of bmodeFrames");
                end
            end
            
            obj.rec.colorRxTangLim = reshape(obj.rec.colorRxTangLim,[],2);
            if obj.rec.colorEnable
                if size(obj.rec.colorRxTangLim,1) == 1
                    obj.rec.colorRxTangLim = obj.rec.colorRxTangLim.*ones(numel(obj.rec.colorFrames),1);
                elseif size(obj.rec.colorRxTangLim,1) ~= numel(obj.rec.colorFrames)
                    error("setRecParams: number of rows in colorRxTangLim must equal the length of colorFrames");
                end
            end
            
            obj.rec.vect0RxTangLim = reshape(obj.rec.vect0RxTangLim,[],2);
            if obj.rec.vectorEnable
                if size(obj.rec.vect0RxTangLim,1) == 1
                    obj.rec.vect0RxTangLim = obj.rec.vect0RxTangLim.*ones(numel(obj.rec.vect0Frames),1);
                elseif size(obj.rec.vect0RxTangLim,1) ~= numel(obj.rec.vect0Frames)
                    error("setRecParams: number of rows in vector0RxTangLim must equal the length of vector0Frames");
                end
            end
            
            obj.rec.vect1RxTangLim = reshape(obj.rec.vect1RxTangLim,[],2);
            if obj.rec.vectorEnable
                if size(obj.rec.vect1RxTangLim,1) == 1
                    obj.rec.vect1RxTangLim = obj.rec.vect1RxTangLim.*ones(numel(obj.rec.vect1Frames),1);
                elseif size(obj.rec.vect1RxTangLim,1) ~= numel(obj.rec.vect1Frames)
                    error("setRecParams: number of rows in vector1RxTangLim must equal the length of vector1Frames");
                end
            end
            
            %% Resulting parameters
            obj.rec.zSize	= length(obj.rec.zGrid);
            obj.rec.xSize	= length(obj.rec.xGrid);
            
            if obj.rec.colorEnable || obj.rec.vectorEnable
                [~,obj.rec.wcFiltInitCoeff] = filter(obj.rec.wcFiltB,obj.rec.wcFiltA,ones(1000,1));
            end
            
            %% If GPU is available...
            obj.rec.gpuEnable	= license('test', 'Distrib_Computing_Toolbox') && ~isempty(ver('parallel')) && parallel.gpu.GPUDevice.isAvailable;
            
            if obj.rec.gpuEnable && obj.rec.gridModeEnable
                % Add location of the CUDA kernels
                addpath([fileparts(mfilename('fullpath')) '\mexcuda']);
                
                % move reconstruction-related data to GPU
                obj.sys.zElem          = gpuArray(single(obj.sys.zElem));
                obj.sys.xElem          = gpuArray(single(obj.sys.xElem));
                obj.sys.tangElem       = gpuArray(single(obj.sys.tangElem));
                obj.rec.zGrid          = gpuArray(single(obj.rec.zGrid));
                obj.rec.xGrid          = gpuArray(single(obj.rec.xGrid));
                obj.rec.rxApod         = gpuArray(single(obj.rec.rxApod));
                obj.seq.txFoc          = gpuArray(single(obj.seq.txFoc));
                obj.seq.txAngZX        = gpuArray(single(obj.seq.txAngZX));
                obj.seq.txApCentZ      = gpuArray(single(obj.seq.txApCentZ));
                obj.seq.txApCentX      = gpuArray(single(obj.seq.txApCentX));
                obj.seq.txFreq         = gpuArray(single(obj.seq.txFreq));
                obj.seq.initDel        = gpuArray(single(obj.seq.initDel));
                obj.seq.txApFstElem    = gpuArray( int32(obj.seq.txApFstElem - 1));
                obj.seq.txApLstElem    = gpuArray( int32(obj.seq.txApLstElem - 1));
                obj.seq.rxApOrig       = gpuArray( int32(obj.seq.rxApOrig - 1));
                obj.seq.nSampOmit      = gpuArray( int32(obj.seq.nSampOmit));
                obj.rec.bmodeRxTangLim = gpuArray(single(obj.rec.bmodeRxTangLim));
                obj.rec.colorRxTangLim = gpuArray(single(obj.rec.colorRxTangLim));
                obj.rec.vect0RxTangLim = gpuArray(single(obj.rec.vect0RxTangLim));
                obj.rec.vect1RxTangLim = gpuArray(single(obj.rec.vect1RxTangLim));
                obj.seq.rxSampFreq     =          single(obj.seq.rxSampFreq);
                obj.rec.sos            =          single(obj.rec.sos);
                obj.seq.startSample    =          single(obj.seq.startSample);
                obj.seq.txDelCent      =          single(obj.seq.txDelCent);
                
            end
        end
        
        function calcTxRxApMask(obj)
            % calcTxRxApMask appends the following fields to the in/out obj:
            % obj.seq.txApOrig      - [element] (1 x nTx) number of probe element being the origin of the tx aperture
            % obj.seq.rxApOrig      - [element] (1 x nTx) number of probe element being the origin of the rx aperture
            % obj.seq.txApFstElem	- [element] (1 x nTx) number of probe element being the first in the tx aperture
            % obj.seq.txApLstElem	- [element] (1 x nTx) number of probe element being the last in the tx aperture
            % obj.seq.txApMask      - [logical] (nElem x nTx) tx aperture mask
            % obj.seq.rxApMask      - [logical] (nElem x nTx) rx aperture mask
            % obj.seq.rxApPadding   - [element] (2 x nTx) rx aperture padding
            
            nElem = obj.sys.nElem;
            iElem = 1:nElem;
            
            obj.seq.txApOrig = round(obj.seq.txCentElem - (obj.seq.txApSize-1)/2 + 1e-9);
            obj.seq.rxApOrig = round(obj.seq.rxCentElem - (obj.seq.rxApSize-1)/2 + 1e-9);
            
            obj.seq.txApFstElem = max(1,     obj.seq.txApOrig);
            obj.seq.txApLstElem = min(nElem, obj.seq.txApOrig + obj.seq.txApSize - 1);
            
            obj.seq.txApMask = (iElem.' >= obj.seq.txApOrig) & (iElem.' <= obj.seq.txApOrig + obj.seq.txApSize - 1);
            obj.seq.rxApMask = (iElem.' >= obj.seq.rxApOrig) & (iElem.' <= obj.seq.rxApOrig + obj.seq.rxApSize - 1);
            
            obj.seq.rxApPadding = [-min(0, obj.seq.rxApOrig - 1); ...
                                    max(0, obj.seq.rxApOrig - 1 + obj.seq.rxApSize - obj.sys.nElem)];
        end

        function calcTxDelays(obj)
            % calcTxDelays appends the following fields to the in/out obj:
            % obj.seq.txDel         - [s] (nElem x nTx) tx delays for each element
            % obj.seq.txDelCent     - [s] (1 x nTx) tx delays for tx aperture centers
            
            %% CALCULATE DELAYS
            txDel = nan(obj.sys.nElem,obj.seq.nTx);
            txDelCent = nan(1,obj.seq.nTx);
            
            isFocInf = isinf(obj.seq.txFoc);
            
            if any(isFocInf)
                % Delays due to the tilting the plane wavefront
                txDel(:,isFocInf)   = (obj.sys.xElem.'             .* sin(obj.seq.txAngZX(isFocInf)) + ...
                                       obj.sys.zElem.'             .* cos(obj.seq.txAngZX(isFocInf))) / obj.seq.c;  % [s] (nElem x nTx) delays for tx elements
                txDelCent(isFocInf)	= (obj.seq.txApCentX(isFocInf) .* sin(obj.seq.txAngZX(isFocInf)) + ...
                                       obj.seq.txApCentZ(isFocInf) .* cos(obj.seq.txAngZX(isFocInf))) / obj.seq.c;  % [s] (1 x nTx) delays for tx aperture center
            end
            
            if any(~isFocInf)
                % Focal point positions
                xFoc = obj.seq.txApCentX(~isFocInf) + obj.seq.txFoc(~isFocInf) .* sin(obj.seq.txAngZX(~isFocInf));  % [m] (1 x nTxFoc) x-position of the focal point
                zFoc = obj.seq.txApCentZ(~isFocInf) + obj.seq.txFoc(~isFocInf) .* cos(obj.seq.txAngZX(~isFocInf));  % [m] (1 x nTxFoc) z-position of the focal point
                
                % Delays due to the element - focal point distances
                txDel(:,~isFocInf)   = sqrt( (xFoc - obj.sys.xElem.'             ).^2 + ...
                                             (zFoc - obj.sys.zElem.'             ).^2) / obj.seq.c; % [s] (nElem x nTx) delays for tx elements
                txDelCent(~isFocInf) = sqrt( (xFoc - obj.seq.txApCentX(~isFocInf)).^2 + ...
                                             (zFoc - obj.seq.txApCentZ(~isFocInf)).^2) / obj.seq.c; % [s] (1 x nTx) delays for tx aperture center
                
                % Inverse the delays for the 'focusing' option (txFoc>0)
                % For 'defocusing' the delays remain unchanged
                focDefoc = 1 - 2 * double(obj.seq.txFoc(~isFocInf)>0);
                txDel(:,~isFocInf)   = txDel(:,~isFocInf)   .* focDefoc;
                txDelCent(~isFocInf) = txDelCent(~isFocInf) .* focDefoc;
            end

            %% Postprocess the delays
            % Make delays = nan outside the tx aperture
            txDel(~obj.seq.txApMask)	= nan;

            % Make delays >= 0 in the tx aperture
            txDelShift	= - min(txDel,[],'omitnan');	% [s] (1 x nTx)
            txDel       = txDel     + txDelShift;       % [s] (nElem x nTx)
            txDelCent	= txDelCent + txDelShift;       % [s] (1 x nTx)

            % Equalize the txDelCent
            txDel       = txDel - txDelCent + max(txDelCent);
            txDelCent	= max(txDelCent);

            % Remove nans
            txDel(~obj.seq.txApMask)	= 0;

            %% Save the delays to the obj
            obj.seq.txDel       = txDel;
            obj.seq.txDelCent	= txDelCent;

        end
        
        function programHW(obj)
            
            import arrus.ops.us4r.*;
            
            obj.us4r.setVoltage(obj.seq.txVoltage);
            
            % Tx/Rx sequence
            nTx = obj.seq.nTx;
            for iTx=1:nTx
                pulse = arrus.ops.us4r.Pulse('centerFrequency', obj.seq.txFreq(iTx), "nPeriods", obj.seq.txNPer(iTx), "inverse", obj.seq.txInvert(iTx));
                txObj = Tx("aperture", obj.seq.txApMask(:,iTx).', 'delays', obj.seq.txDel(:,iTx).', "pulse", pulse);
                rxObj = Rx("aperture", obj.seq.rxApMask(:,iTx).', "padding", obj.seq.rxApPadding(:,iTx).', "sampleRange", obj.seq.startSample + [0, obj.seq.nSamp], "downsamplingFactor", obj.seq.fpgaDec);
                txrxList(iTx) = TxRx("tx", txObj, "rx", rxObj, "pri", obj.seq.txPri);
            end
            txrxSeq = TxRxSequence("ops", txrxList, "nRepeats", obj.seq.nRep, "tgcCurve", obj.seq.tgcCurve);
            
            % Digital Down Conversion
            if obj.seq.hwDdcEnable
                ddc = DigitalDownConversion( ...
                    "demodulationFrequency", mean(obj.seq.txFreq), ...
                    "decimationFactor", obj.seq.dec, ...
                    "firCoefficients", obj.seq.ddcFirCoeff);
            else
                ddc = [];
            end
            
            % Upload scheme
            scheme = Scheme('txRxSequence', txrxSeq, 'workMode', "MANUAL", 'digitalDownConversion', ddc);
            
            [obj.buffer.data, ...
             obj.buffer.framesOffset, ...
             obj.buffer.framesNumber, ...
             obj.buffer.oemId, ...
             obj.buffer.frameId, ...
             obj.buffer.channelId] = obj.session.upload(scheme);
            
            % Data reorganization addresses
            nChan = obj.sys.nChArius;
            nRep = obj.seq.nRep;
            iRep = uint32(reshape(0:(nRep-1),1,1,nRep));
            
            obj.buffer.reorgAddrDest = (1 : obj.seq.rxApSize*nTx*nRep).';
            obj.buffer.reorgAddrOrig = 1 + ...
                ( obj.buffer.framesOffset(1 + obj.buffer.oemId) + ...                 % offset due to oemId
                  obj.buffer.framesNumber(1 + obj.buffer.oemId) / nRep .* iRep + ...  % offset due to iRep
                  obj.buffer.frameId ) * nChan  + ...                                 % offset due to frameId
                uint32(obj.buffer.channelId);                                         % offset due to channelId

            obj.buffer.reorgAddrDest = obj.buffer.reorgAddrDest(repmat(obj.buffer.channelId,1,1,nRep) >= 0);
            obj.buffer.reorgAddrOrig = obj.buffer.reorgAddrOrig(repmat(obj.buffer.channelId,1,1,nRep) >= 0);
        end

        function [rf, metadata] = execSequence(obj)

            if ~obj.sys.isHardwareProgrammed
                error("execSequence: hardware is not programmed, sequence cannot be executed");
            end
            
            nChan	= obj.sys.nChArius;
            nSamp	= obj.seq.nSamp;
            nTx     = obj.seq.nTx;
            nRep	= obj.seq.nRep;
            nTrig0  = obj.buffer.framesNumber(1);

            %% Capture & transfer data to PC
            obj.session.run();
            rf0 = obj.buffer.data.front().eval();
            
            %% Get metadata
            metadata = zeros(nChan, nTrig0, 'int16');
            metadata(:, :) = rf0(:, 1:nSamp:nTrig0*nSamp);

            %% Reorganize
            if obj.seq.hwDdcEnable
                rf0 = reshape(rf0, nChan, 2, nSamp, sum(obj.buffer.framesNumber));
                rf0 = complex(rf0(:,1,:,:), rf0(:,2,:,:));
            end
            rf0 = reshape(rf0, nChan, nSamp, sum(obj.buffer.framesNumber));
            rf0 = permute(rf0, [2 1 3]);
            
            rf  = zeros(nSamp, obj.seq.rxApSize*nTx*nRep,'like',rf0);
            rf(:,obj.buffer.reorgAddrDest) = rf0(:, obj.buffer.reorgAddrOrig);
            rf  = reshape(rf, nSamp, obj.seq.rxApSize, nTx, nRep);

        end
        
        function img = execReconstr(obj,rfRaw)

            %% Move data to GPU if possible
            if obj.rec.gpuEnable
                rfRaw = gpuArray(rfRaw);
            end
            
            rfRaw = double(rfRaw);

            %% Preprocessing
            % Raw rf data filtration
            if obj.rec.filtEnable
                rfRaw = filter(obj.rec.filtB,obj.rec.filtA,rfRaw);
            end

            rfRaw = single(rfRaw);
            
            % Digital Down Conversion
            if obj.rec.swDdcEnable
                rfRaw = downConversion(rfRaw,obj.seq,obj.rec);
            end

            %% Reconstruction
            if ~obj.rec.gridModeEnable
                rfBfr = reconstructRfLin(rfRaw,obj.sys,obj.seq,obj.rec);
            else
%                 rfBfr = reconstructRfImg(rfRaw,obj.sys,obj.seq,obj.rec);
                
                % B-Mode image reconstruction
                if obj.rec.bmodeEnable
                    rfBfr = obj.runCudaReconstruction(rfRaw,'bmode');
                    
                    if obj.sys.interfEnable
                        ccf = 1 - sqrt( var(real(rfBfr)./abs(rfBfr), 0, 3) + ...
                                        var(imag(rfBfr)./abs(rfBfr), 0, 3) );
                        rfBfr = rfBfr .* ccf;
                        
                        % coherent compounding for NDT
                        rfBfr = mean(rfBfr,3,'omitnan');
                    else
                        % incoherent compounding for medical imaging
                        rfBfr = mean(abs(rfBfr),3,'omitnan');
                    end
                    
                end
                
                % Color Doppler image reconstruction
                if obj.rec.colorEnable
                    rfBfrColor = obj.runCudaReconstruction(rfRaw,'color');
                    
                    [color,power] = dopplerColorImaging(rfBfrColor, obj.seq, obj.rec);
                end
                
                % Vector Doppler image reconstruction
                if obj.rec.vectorEnable
                    rfBfrVect0 = obj.runCudaReconstruction(rfRaw,'vector0');
                    rfBfrVect1 = obj.runCudaReconstruction(rfRaw,'vector1');
                    
                    [color,power] = dopplerColorImaging(cat(4,rfBfrVect0,rfBfrVect1), obj.seq, obj.rec);
                end
            end

            %% Postprocessing
            % Obtain complex signal (if it isn't complex already)
            if ~obj.seq.hwDdcEnable && ~obj.rec.swDdcEnable
                nanMask = isnan(rfBfr);
                rfBfr(nanMask) = 0;
                rfBfr = hilbert(rfBfr);
                rfBfr(nanMask) = nan;
            end
            
            % Envelope detection
            envImg = abs(rfBfr);
            
            % Scan conversion
            if ~obj.rec.gridModeEnable
                envImg = scanConversion(envImg,obj.sys,obj.seq,obj.rec);
                
                % Doppler is not implemented for 'lin' mode
                % NDT interface is not supported in scanConversion
            end
            
            % Compression
            img = 20*log10(envImg);
            
            if obj.rec.colorEnable || obj.rec.vectorEnable
                power = 10*log10(power);
            end
            
            % Put B-Mode & Doppler data together
            if obj.rec.colorEnable || obj.rec.vectorEnable
                img = cat(4,img,power,color);
            end
            
            % Gather data from GPU
            if obj.rec.gpuEnable
                img = gather(img);
            end
            
            
        end
        
        function iqLri = runCudaReconstruction(obj,iqRaw,selFramesType)
            
            switch selFramesType
                case 'bmode'
                    selFrames = obj.rec.bmodeFrames;
                    rxTangLim = obj.rec.bmodeRxTangLim;
                case 'color'
                    selFrames = obj.rec.colorFrames;
                    rxTangLim = obj.rec.colorRxTangLim;
                case 'vector0'
                    selFrames = obj.rec.vect0Frames;
                    rxTangLim = obj.rec.vect0RxTangLim;
                case 'vector1'
                    selFrames = obj.rec.vect1Frames;
                    rxTangLim = obj.rec.vect1RxTangLim;
                otherwise
                    error('runCudaReconstruction: invalid modality name.');
            end
            
            if ~obj.sys.interfEnable
                iqLri	= iqRaw2Lri(iqRaw(:,:,selFrames), ...
                                    obj.sys.zElem, ...
                                    obj.sys.xElem, ...
                                    obj.sys.tangElem, ...
                                    obj.rec.zGrid, ...
                                    obj.rec.xGrid, ...
                                    obj.rec.rxApod, ...
                                    obj.seq.txFoc(selFrames), ...
                                    obj.seq.txAngZX(selFrames), ...
                                    obj.seq.txApCentZ(selFrames), ...
                                    obj.seq.txApCentX(selFrames), ...
                                    obj.seq.txFreq(selFrames), ...
                                    obj.seq.initDel(selFrames), ...
                                    obj.seq.txApFstElem(selFrames), ...
                                    obj.seq.txApLstElem(selFrames), ...
                                    obj.seq.rxApOrig(selFrames), ...
                                    obj.seq.nSampOmit(selFrames)/obj.rec.dec, ...
                                    rxTangLim(:,1).', ...
                                    rxTangLim(:,2).', ...
                                    obj.seq.rxSampFreq/obj.rec.dec, ...
                                    obj.rec.sos);
            else
                iqLri	= iqRaw2Lri_SSTA_Wedge( ...
                                    iqRaw(:,:,selFrames), ...
                                    obj.sys.zElem, ...
                                    obj.sys.xElem, ...
                                    obj.sys.tangElem, ...
                                    obj.rec.zGrid, ...
                                    obj.rec.xGrid, ...
                                    obj.seq.txApCentZ(selFrames), ...
                                    obj.seq.txApCentX(selFrames), ...
                                    obj.seq.rxApOrig(selFrames), ...
                                    gather(rxTangLim(1,1)), ...
                                    gather(rxTangLim(1,2)), ...
                                    obj.seq.rxSampFreq/obj.rec.dec, ...
                                    gather(obj.seq.txFreq(1)), ...
                                    obj.rec.sos, ...
                                    obj.sys.interfSos, ...
                                    1/64/gather(obj.seq.txFreq(1)), ...
                                    gather(obj.seq.initDel(1)));
            end
            
        end
        
    end
end
