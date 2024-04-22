classdef Us4R < handle
    % A handle to the Us4R system. 
    %
    % This class provides functions to configure the system and perform
    % data acquisition using the Us4R.
    % 
    % :param configFile: name of the prototxt file containing setup information.
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
            addParameter(paramsParser, 'logTime', false, @(x) validateattributes(x, {'logical'}, {'scalar'}, 'Us4R', 'logTime'));
            parse(paramsParser, varargin{:});
            
            configFile   = paramsParser.Results.configFile;
            interfEnable = paramsParser.Results.interfEnable;
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
            obj.sys.maxSeqLength = 2^14;
            obj.sys.adcVolt2Lsb = (2^16)/2; % 16-bit coding of 2Vpp range
            obj.sys.tgcOffset = 359; % [samp] includes tgcTriggerOffset=211 and tgcHalfResponseOffset=148;
            obj.sys.tgcInterv = 153; % [samp]
            obj.logTime = logTime;
            
            % Check if valid GPU is available
            isGpuAvailable = license('test', 'Distrib_Computing_Toolbox') ...
                           && ~isempty(ver('parallel')) ...
                           && parallel.gpu.GPUDevice.isAvailable;
            if ~isGpuAvailable
                error('Arrus requires Parallel Computing Toolbox and a supported GPU device');
            end
            
            % Probe parameters
            probe = obj.us4r.getProbeModel;
            obj.sys.nElem = double(probe.nElements);
            obj.sys.pitch = probe.pitch;
            obj.sys.freqRange = double(probe.txFrequencyRange);
            obj.sys.curvRadius = -probe.curvatureRadius; % (-/+ for convex/concave probes)

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
        
        function stopScheme(obj)
            obj.session.stopScheme();
        end
        
        function nProbeElem = getNProbeElem(obj)
            nProbeElem = obj.sys.nElem;
        end

        function setLnaGain(obj,gain)
            obj.us4r.setLnaGain(gain);
        end

        function setPgaGain(obj,gain)
            obj.us4r.setPgaGain(gain);
        end

        function setTgcCurve(varargin)
            obj = varargin{1};
            obj.us4r.setTgcCurve(varargin{2:end});
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
                'txInvert', sequenceOperation.txInvert, ...
                'workMode', sequenceOperation.workMode, ...
                'sri', sequenceOperation.sri, ...
                'bufferSize', sequenceOperation.bufferSize);
            
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
                'wcFiltInitSize', reconstructOperation.wcFiltInitSize, ...
                'cohFiltEnable', reconstructOperation.cohFiltEnable, ...
                'cohCompEnable', reconstructOperation.cohCompEnable);
            
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
            obj.session.stopScheme();

            rf = obj.rawDataReorganization(rf);
            
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
            obj.session.stopScheme();
            
            rf = obj.rawDataReorganization(rf);

            if obj.rec.enable
                img = obj.execReconstr(rf(:,:,:,1));
            else
                img = [];
            end
        end
        
        function [rawBuffer, imgBuffer, sriBuffer] = runLoop(obj, isContinue, callback, varargin)
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
            % :param bufferType: type of data stored in the cineloop buffer, \
            %   can be "none", "raw", "img", or "all" (optional, default="none").
            % :param bufferMode: buffer mode of operation. Can be "conc" \
            %   for concurrent or "subs" for subsequent operation with the \
            %   callback function. For "conc" the buffer is used as long as \
            %   isContinue is true. For "subs" the buffer is used as soon as \
            %   isContinue is false until the buffer is full.
            % :param bufferSize: size of the cineloop buffer as a number \
            %   of sequence executions (optional, default=1).
            %
            % :returns: buffers containing raw data (rawBuffer), image data \
            %   (imgBuffer), and sequence repetition intervals (sriBuffer).
            
            % Input parser
            paramsParser = inputParser;
            addParameter(paramsParser, 'bufferType', 'none', @(x) validateattributes(x, {'char','string'}, {'scalartext'}, 'runLoop', 'bufferType'));
            addParameter(paramsParser, 'bufferMode', 'conc', @(x) validateattributes(x, {'char','string'}, {'scalartext'}, 'runLoop', 'bufferMode'));
            addParameter(paramsParser, 'bufferSize', 1,      @(x) validateattributes(x, {'numeric'}, {'scalar','integer','real','positive','finite'}, 'runLoop', 'bufferSize'));
            parse(paramsParser, varargin{:});
            
            bufferType = paramsParser.Results.bufferType;
            bufferMode = paramsParser.Results.bufferMode;
            bufferSize = paramsParser.Results.bufferSize;
            
            if ~any(strcmp(bufferType,{'none','raw','img','all'}))
                warning('runLoop: bufferType must be one of the following: "none", "raw", "img", or "all". Buffer disabled.');
                bufferType = 'none';
            end
            
            if ~any(strcmp(bufferMode,{'conc','subs'}))
                warning('runLoop: bufferMode must be one of the following: "conc" or "subs". Buffer disabled.');
                bufferType = 'none';
            end

            if strcmp(bufferType,'none')
                bufferMode = 'conc';
            end
            
            rawBufferEnable = any(strcmp(bufferType,{'raw','all'}));
            imgBufferEnable = any(strcmp(bufferType,{'img','all'}));
            if imgBufferEnable && ~obj.rec.enable
                warning('runLoop: Reconstruction must be enabled to acquire image data. Image buffer disabled.');
                imgBufferEnable = false;
            end
            
            concBufferEnable = strcmp(bufferMode,'conc');
            
            % Buffers initialization
            sriBuffer = nan(bufferSize,1);
            
            if ~concBufferEnable
                sampSize = 1 + double(obj.seq.hwDdcEnable);
                raw0Buffer = zeros(obj.sys.nChArius, sum(obj.buffer.framesNumber) * obj.seq.nSamp * sampSize, bufferSize, 'int16');
            end
                
            if rawBufferEnable
                rawBuffer = zeros(obj.seq.nSamp, obj.seq.rxApSize, obj.seq.nTx, obj.seq.nRep, bufferSize, 'single');
                if obj.seq.hwDdcEnable
                    rawBuffer = complex(rawBuffer,0);
                end
            else
                rawBuffer = [];
            end
            
            if imgBufferEnable
                nBufferLayers = 1 + 3*double(obj.rec.colorEnable && ~obj.rec.vectorEnable) + 4*double(obj.rec.vectorEnable);
                imgBuffer = nan(obj.rec.zSize, obj.rec.xSize, bufferSize, nBufferLayers, 'single');
            else
                imgBuffer = [];
            end
            
            % Main loop (using buffers if bufferMode is "conc")
            i = 0;
            tStampPrev = nan;
            while(isContinue())
                i = i + 1;
                
                tic;
                [rf, meta] = obj.execSequence;
                acqTime = toc;
                
                tic;
                rf = obj.rawDataReorganization(rf);
                reorgTime = toc;
                
                tStampCurr = bin2dec(reshape(dec2bin(meta([8 7 6 5]),16).',1,64)) / obj.sys.rxSampFreq; % [s]
                sri = tStampCurr - tStampPrev;
                tStampPrev = tStampCurr;
                
                if obj.rec.enable
                    tic;
                    img = obj.execReconstr(rf(:,:,:,1));
                    recTime = toc;
                    callback(img);
                else
                    callback(rf);
                end
                
                % Log time
                if obj.logTime
                    disp(['Frame no. ' num2str(i)]);
                    disp(['Acq.  time = ' num2str(acqTime, '%5.3f') ' s']);
                    disp(['Reorg.  time = ' num2str(reorgTime, '%5.3f') ' s']);
                    if exist('recTime', 'var')
                        disp(['Rec.  time = ' num2str(recTime, '%5.3f') ' s']);
                    end
                    disp(['Frame rate = ' num2str(1/sri, '%5.1f') ' fps']);
                    disp('--------------------');
                end
                
                % Copy data to buffers
                if concBufferEnable
                    I = mod(i-1,bufferSize)+1;
                    
                    if rawBufferEnable
                        rawBuffer(:,:,:,:,I) = rf;
                    end
                    
                    if imgBufferEnable
                        imgBuffer(:,:,I,:) = img;
                    end
                    
                    sriBuffer(I) = sri;
                end
            end

            if concBufferEnable
                obj.session.stopScheme();
                disp('runLoop: acquisition done');
                
                % Output buffer unwinding
                rawBuffer = circshift(rawBuffer,-I,5);
                imgBuffer = circshift(imgBuffer,-I,3);
                sriBuffer = circshift(sriBuffer,-I,1);
                
                disp('runLoop: postprocessing done');
            else
                % Second loop (acquisition of raw data for "subs" bufferMode)
                for i=1:bufferSize
                    [raw0Buffer(:,:,i), meta] = obj.execSequence;
                    
                    tStampCurr = bin2dec(reshape(dec2bin(meta([8 7 6 5]),16).',1,64)) / obj.sys.rxSampFreq; % [s]
                    sriBuffer(i) = tStampCurr - tStampPrev;
                    tStampPrev = tStampCurr;
                end
                obj.session.stopScheme();
                disp('runLoop: acquisition done');

                % Postprocessing loop
                for i=1:bufferSize
                    rf = obj.rawDataReorganization(raw0Buffer(:,:,i));
                    
                    if rawBufferEnable
                        rawBuffer(:,:,:,:,i) = rf;
                    end
                    
                    if imgBufferEnable
                        imgBuffer(:,:,i,:) = obj.execReconstr(rf(:,:,:,1));
                    end
                end
                disp('runLoop: postprocessing done');
            end
        end
        
        function plotRawRf(obj,varargin)
            % :param selectedLines: vector indicating the rf lines to be displayed. \
            %   Rf lines are numbered as follows: 1:rxApertureSize*nTx*nRep. \
            %   (optional name-value argument)
            % :param amplitudeLim: scalar defining the displayed amplitude range \
            %   as [-amplitudeLim, amplitudeLim] (optional name-value argument)
            % :param boundsModeEnable: logical scalar determining if min and max \
            %   values of each sample from a set of rf lines are displayed \
            %   instead of individual rf lines. (optional name-value argument)
            % :param linRangeEnable: logical scalar determining if range of \
            %   undistorted amplitudes is displayed. (optional name-value argument)
            
            %% Check if system is ready for plotRawRf execution
            if ~obj.sys.isHardwareProgrammed
                error("plotRawRf: hardware is not programmed, rf cannot be collected");
            end
            if obj.seq.hwDdcEnable
                error("plotRawRf: hardware DDC is enabled, rf cannot be collected");
            end
            
            %% Input parser
            dispParParser = inputParser;
            
            nLine = obj.seq.rxApSize * obj.seq.nTx * obj.seq.nRep;
            addParameter(dispParParser, 'selectedLines', 1:nLine, ...
                @(x) assert(isvector(x) && isnumeric(x) && all(x>0), ...
                "selectedLines must be a positive numerical vector."));
            
            addParameter(dispParParser, 'amplitudeLim', 2^15, ...
                @(x) assert(isscalar(x) && isnumeric(x) && x>0, ...
                "amplitudeLim must be a positive numerical scalar."));
            
            addParameter(dispParParser, 'boundsModeEnable', false, ...
                @(x) assert(isscalar(x) && islogical(x), ...
                "boundsModeEnable must be a logical scalar."));
            
            addParameter(dispParParser, 'linRangeEnable', false, ...
                @(x) assert(isscalar(x) && islogical(x), ...
                "linRangeEnable must be a logical scalar."));
            
            parse(dispParParser, varargin{:});
            
            selectedLines = dispParParser.Results.selectedLines;
            amplitudeLim  = dispParParser.Results.amplitudeLim;
            boundsEnable  = dispParParser.Results.boundsModeEnable;
            linRngEnable  = dispParParser.Results.linRangeEnable;
            
            %% Prepare figure
            nSamp = obj.seq.nSamp;
            if boundsEnable
                nLine = 2;
            else
                nLine = numel(selectedLines);
            end
            
            % Create figure.
            hFig = figure();
            hDisp = plot(nan(nSamp,nLine), 1:nSamp);
            xlabel('amplitude');
            ylabel('sample #');
            set(gca,'XLim', amplitudeLim*[-1 1]);
            set(gca,'YLim', [0 nSamp+1]);
            set(gca,'YDir', 'reverse');
            grid on;

            % Undistorted amplitude range
            if linRngEnable
                switch obj.us4r.getLnaGain
                    case 12, voltLim = 0.243; %[V]
                    case 18, voltLim = 0.219; %[V]
                    case 24, voltLim = 0.152; %[V]
                    otherwise
                        error('Unsupported LNA gain value.');
                end
                
                tgcCurveResamp = interp1(obj.seq.tgcPoints, obj.seq.tgcCurve, ...
                                         (obj.seq.startSample + (1:nSamp) - 1)*obj.seq.dec, "linear", nan);
                ampUndistortLim = voltLim * 10.^(tgcCurveResamp/20) * obj.sys.adcVolt2Lsb;
                ampUndistortLim = min(ampUndistortLim, 2^15, "includenan");
                
                hold on;
                plot(ampUndistortLim(:).*[-1 1], 1:nSamp,'Color','k','LineStyle',':','LineWidth',2);
            end
            
            while(ishghandle(hFig))
                data = obj.run;
                data = data(:,selectedLines);
                if boundsEnable
                    data = [min(data,[],2), max(data,[],2)];
                end

                try
                    for iLine=1:nLine
                        set(hDisp(iLine), 'XData', data(:,iLine));
                    end
                    drawnow limitrate;
                    
                catch ME
                    if(strcmp(ME.identifier, 'MATLAB:class:InvalidHandle'))
                        disp('Display was closed.');
                    else
                        rethrow(ME);
                    end
                end
            end
        end
        
        function imageRawRf(obj,varargin)
            % :param selectedLines: vector indicating the rf lines to be displayed. \
            %   Rf lines are numbered as follows: 1:rxApertureSize*nTx*nRep. \
            %   (optional name-value argument)
            % :param amplitudeLim: scalar defining the displayed amplitude range \
            %   as [-amplitudeLim, amplitudeLim] (optional name-value argument)
            
            %% Check if system is ready for plotRawRf execution
            if ~obj.sys.isHardwareProgrammed
                error("plotRawRf: hardware is not programmed, rf cannot be collected");
            end
            if obj.seq.hwDdcEnable
                error("plotRawRf: hardware DDC is enabled, rf cannot be collected");
            end
            
            %% Input parser
            dispParParser = inputParser;
            
            nLine = obj.seq.rxApSize * obj.seq.nTx * obj.seq.nRep;
            addParameter(dispParParser, 'selectedLines', 1:nLine, ...
                @(x) assert(isvector(x) && isnumeric(x) && all(x>0), ...
                "selectedLines must be a positive numerical vector."));
            
            addParameter(dispParParser, 'amplitudeLim', 2^15, ...
                @(x) assert(isscalar(x) && isnumeric(x) && x>0, ...
                "amplitudeLim must be a positive numerical scalar."));
            
            parse(dispParParser, varargin{:});
            
            selectedLines = dispParParser.Results.selectedLines;
            amplitudeLim  = dispParParser.Results.amplitudeLim;
            
            %% Prepare figure
            nSamp = obj.seq.nSamp;
            nLine = numel(selectedLines);
            
            % Create figure.
            hFig = figure();
            hDisp = imagesc(1:nLine, 1:nSamp, []);
            xlabel('rf line #');
            ylabel('sample #');
            set(gca,'XLim', [0.5 nLine+0.5]);
            set(gca,'YLim', [0.5 nSamp+0.5]);
            set(gca,'CLim', amplitudeLim*[-1 1]);
            colormap(jet);
            colorbar;
            
            while(ishghandle(hFig))
                data = obj.run;
                try
                    set(hDisp, 'CData', data(:,selectedLines));
                    drawnow limitrate;
                    
                catch ME
                    if(strcmp(ME.identifier, 'MATLAB:class:InvalidHandle'))
                        disp('Display was closed.');
                    else
                        rethrow(ME);
                    end
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
                                'txInvert',         'txInvert'; ...
                                'workMode',         'workMode'; ...
                                'sri',              'sri'; ...
                                'bufferSize',       'bufferSize'};

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

            %% Validate buffer size
            if obj.seq.bufferSize < 2
                error("setSeqParams: bufferSize must be >= 2");
            end
            
            if obj.seq.bufferSize * obj.seq.nTx > obj.sys.maxSeqLength
                error("setSeqParams: product of bufferSize and sequence length cannot exceed " ...
                    + num2str(obj.sys.maxSeqLength));
            end
            
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
            txPriMin = (obj.seq.startSample + obj.seq.nSamp) / obj.seq.rxSampFreq + 42e-6;
            if isempty(obj.seq.txPri)
                obj.seq.txPri = txPriMin;
            elseif obj.seq.txPri < txPriMin
                warning(['txPri value is too low. It is increased to ' num2str(txPriMin)]);
                obj.seq.txPri = txPriMin;
            end
            
            %% TGC
            obj.seq.tgcLim = double(obj.us4r.getLnaGain + obj.us4r.getPgaGain) + [-40 0];
            
            % Default TGC start level
            if isempty(obj.seq.tgcStart)
                obj.seq.tgcStart = obj.seq.tgcLim(1);
            end
            
            obj.seq.tgcPoints = obj.sys.tgcOffset : obj.sys.tgcInterv : (obj.seq.startSample + obj.seq.nSamp - 1)*obj.seq.dec; % [samp]
            obj.seq.tgcCurve = obj.seq.tgcStart + obj.seq.tgcSlope * obj.seq.tgcPoints / obj.sys.rxSampFreq * obj.seq.c;  % [dB]
            if any(obj.seq.tgcCurve < obj.seq.tgcLim(1) | obj.seq.tgcCurve > obj.seq.tgcLim(2))
                warning(['For LNA=' num2str(obj.us4r.getLnaGain) ...
                      'dB and PGA=' num2str(obj.us4r.getPgaGain) ...
                      'dB, TGC values are limited to ' num2str(obj.seq.tgcLim(1)) '-'  num2str(obj.seq.tgcLim(2)) 'dB range.']);
                obj.seq.tgcCurve = max(obj.seq.tgcLim(1),min(obj.seq.tgcLim(2),obj.seq.tgcCurve));
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
                                'wcFiltInitSize',   'wcFiltInitSize'; ...
                                'cohFiltEnable',    'cohFiltEnable'; ...
                                'cohCompEnable',    'cohCompEnable'};

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

            %% Radial coordinates for classical reconstruction
            if ~obj.rec.gridModeEnable
                t = (obj.seq.startSample + (0:(obj.seq.nSamp-1))) / obj.seq.rxSampFreq;
                obj.rec.rGrid = t * obj.rec.sos / 2;
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
            
            if (obj.rec.colorEnable || obj.rec.vectorEnable) && ~isempty(obj.rec.wcFiltA)
                [~,obj.rec.wcFiltInitCoeff] = filter(obj.rec.wcFiltB,obj.rec.wcFiltA,ones(1000,1));
            end
            
            %% Move data to GPU...
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
            obj.rec.wcFiltB        = gpuArray(single(obj.rec.wcFiltB));
            obj.rec.wcFiltA        = gpuArray(single(obj.rec.wcFiltA));
            obj.seq.rxSampFreq     =          single(obj.seq.rxSampFreq);
            obj.rec.sos            =          single(obj.rec.sos);
            obj.seq.startSample    =          single(obj.seq.startSample);
            obj.seq.txDelCent      =          single(obj.seq.txDelCent);
            
            if (obj.rec.colorEnable || obj.rec.vectorEnable) && ~isempty(obj.rec.wcFiltA)
                obj.rec.wcFiltInitCoeff = gpuArray(single(obj.rec.wcFiltInitCoeff)).';
            end
            
            if ~obj.rec.gridModeEnable
                obj.rec.rGrid      = gpuArray(single(obj.rec.rGrid));
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
            
            if obj.seq.txVoltage == 0
                obj.us4r.disableHV();
            else
                obj.us4r.setVoltage(obj.seq.txVoltage);
            end
            
            % Tx/Rx sequence
            nTx = obj.seq.nTx;
            for iTx=1:nTx
                pulse = arrus.ops.us4r.Pulse('centerFrequency', obj.seq.txFreq(iTx), "nPeriods", obj.seq.txNPer(iTx), "inverse", obj.seq.txInvert(iTx));
                txObj = Tx("aperture", obj.seq.txApMask(:,iTx).', 'delays', obj.seq.txDel(:,iTx).', "pulse", pulse);
                rxObj = Rx("aperture", obj.seq.rxApMask(:,iTx).', "padding", obj.seq.rxApPadding(:,iTx).', "sampleRange", obj.seq.startSample + [0, obj.seq.nSamp], "downsamplingFactor", obj.seq.fpgaDec);
                txrxList(iTx) = TxRx("tx", txObj, "rx", rxObj, "pri", obj.seq.txPri);
            end
            txrxSeq = TxRxSequence("ops", txrxList, "nRepeats", obj.seq.nRep, "tgcCurve", obj.seq.tgcCurve, "sri", obj.seq.sri);
            
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
            scheme = Scheme('txRxSequence', txrxSeq, ...
                            'workMode', obj.seq.workMode, ...
                            'digitalDownConversion', ddc, ...
                            'rxBufferSize', obj.seq.bufferSize, ...
                            'outputBuffer', arrus.framework.DataBufferDef("type", "FIFO", "nElements", obj.seq.bufferSize));
            
            [obj.buffer.data, ...
             obj.buffer.framesOffset, ...
             obj.buffer.framesNumber, ...
             obj.buffer.oemId, ...
             obj.buffer.frameId, ...
             obj.buffer.channelId] = obj.session.upload(scheme);

            obj.buffer.framesOffset = obj.buffer.framesOffset.';
            obj.buffer.framesNumber = obj.buffer.framesNumber.';

            obj.buffer.iFrame = 0;
            
            % Data reorganization addresses (old)
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

            % Data reorganization addresses (new)
            obj.buffer.framesOffset = double(obj.buffer.framesOffset);
            obj.buffer.framesNumber = double(obj.buffer.framesNumber);
            obj.buffer.oemId        = double(obj.buffer.oemId);
            obj.buffer.frameId      = double(obj.buffer.frameId);
            obj.buffer.channelId    = double(obj.buffer.channelId);

            nOem = numel(obj.buffer.framesNumber);
            nChunk = sum(obj.buffer.framesNumber);
            nChan = obj.sys.nChArius;
            nRep = obj.seq.nRep;
            nRx = obj.seq.rxApSize;
            
            obj.buffer.reorgMap = - ones(nChan, nChunk, 'int32');
            
            for iOem=1:nOem
                nFrame = obj.buffer.framesNumber(iOem) / nRep;
                for iFrame=1:nFrame
                    isSelect = obj.buffer.oemId == iOem-1 ...
                             & obj.buffer.frameId == iFrame-1 ...
                             & obj.buffer.channelId >= 0;
                    iTx = find(any(isSelect));
                    iRx = find(isSelect(:,iTx) & obj.buffer.channelId(:,iTx) >= 0);
                    iChan = obj.buffer.channelId(iRx,iTx) + 1;
                    for iRep=1:nRep
                        iChunk = obj.buffer.framesOffset(iOem) ...
                               + obj.buffer.framesNumber(iOem) / nRep * (iRep-1) ...
                               + iFrame;
                        obj.buffer.reorgMap(iChan,iChunk) = (iRep-1)*nTx*nRx + (iTx-1)*nRx + iRx-1; % 0-based indexing
                    end
                end
            end
            
        end
        
        function [rf, metadata] = execSequence(obj)
            
            if ~obj.sys.isHardwareProgrammed
                error("execSequence: hardware is not programmed, sequence cannot be executed");
            end
            
            %% Capture & transfer data to PC
            if obj.buffer.iFrame == 0 || strcmp(obj.seq.workMode,"MANUAL")
                obj.session.run();
            end
            rf = obj.buffer.data.front().eval();
            rf = rf(:,:);
            
            obj.buffer.iFrame = obj.buffer.iFrame + 1;
            
            %% Get metadata
            nChan	= obj.sys.nChArius;
            nSamp	= obj.seq.nSamp;
            nTrig0  = obj.buffer.framesNumber(1);

            metadata = zeros(nChan, nTrig0, 'int16');   % preallocate memory? Is metadata overlayed on the rf or does it move the rf? Delays!!!
            metadata(:, :) = rf(:, 1:nSamp:nTrig0*nSamp);
            
        end
        
        function img = execReconstr(obj,rfRaw)
            
            %% Move data to GPU if possible
            rfRaw = gpuArray(rfRaw);
            
            %% Preprocessing
            % Raw rf data filtration
            if obj.rec.filtEnable
                rfRaw = double(rfRaw);
                rfRaw = filter(obj.rec.filtB,obj.rec.filtA,rfRaw);
                rfRaw = single(rfRaw);
            end
            
            % Digital Down Conversion
            if obj.rec.swDdcEnable
                rfRaw = downConversion(rfRaw,obj.seq,obj.rec);
            end
            
            %% Reconstruction
            if ~obj.rec.gridModeEnable
                if numel(obj.rec.bmodeFrames) ~= obj.seq.nTx || ...
                   any(obj.rec.bmodeFrames ~= 1:obj.seq.nTx) || ...
                   obj.rec.colorEnable || obj.rec.vectorEnable
                    error("execReconstr: frames selection or doppler modes are not supported when gridModeEnable=false");
                end
                if obj.rec.bmodeEnable
                    rfBfr = obj.runCudaReconstructionLin(rfRaw);
                end
            else
                % B-Mode image reconstruction
                if obj.rec.bmodeEnable
                    rfBfr = obj.runCudaReconstruction(rfRaw,'bmode');
                    
                    % Coherence filtration
                    if obj.rec.cohFiltEnable
                        ccf = 1 - sqrt( var(real(rfBfr)./abs(rfBfr), 0, 3) + ...
                                        var(imag(rfBfr)./abs(rfBfr), 0, 3) );
                        rfBfr = rfBfr .* ccf;
                    end
                    
                    % Coherent/Incoherent compounding
                    if obj.rec.cohCompEnable
                        rfBfr = mean(rfBfr,3,'omitnan');
                    else
                        rfBfr = mean(abs(rfBfr),3,'omitnan');
                    end
                end
                
                % Color Doppler image reconstruction
                if obj.rec.colorEnable
                    rfBfrColor = obj.runCudaReconstruction(rfRaw,'color');
                    
                    [color,power,turbu] = dopplerColorImaging(rfBfrColor, obj.seq, obj.rec);
                end
                
                % Vector Doppler image reconstruction
                if obj.rec.vectorEnable
                    rfBfrVect0 = obj.runCudaReconstruction(rfRaw,'vector0');
                    rfBfrVect1 = obj.runCudaReconstruction(rfRaw,'vector1');
                    
                    [color,power,turbu] = dopplerColorImaging(cat(4,rfBfrVect0,rfBfrVect1), obj.seq, obj.rec);
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
                img = cat(4,img,power,turbu,color);
            end
            
            % Gather data from GPU
            img = gather(img);
            
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
        
        function iqLri = runCudaReconstructionLin(obj,iqRaw)
            
            selFrames = obj.rec.bmodeFrames;
            
            iqLri	= iqRaw2Lin(iqRaw(:,:,selFrames), ...
                                obj.sys.zElem, ...
                                obj.sys.xElem, ...
                                obj.sys.tangElem, ...
                                obj.rec.rGrid, ...
                                obj.rec.rxApod, ...
                                obj.seq.txAngZX(selFrames), ...
                                obj.seq.txApCentZ(selFrames), ...
                                obj.seq.txApCentX(selFrames), ...
                                obj.seq.txFreq(selFrames), ...
                                obj.seq.initDel(selFrames), ...
                                obj.seq.rxApOrig(selFrames), ...
                                obj.seq.nSampOmit(selFrames)/obj.rec.dec, ...
                                obj.rec.bmodeRxTangLim(:,1).', ...
                                obj.rec.bmodeRxTangLim(:,2).', ...
                                obj.seq.rxSampFreq/obj.rec.dec, ...
                                obj.rec.sos);
            
        end
        
        function [dataOut] = rawDataReorganization(obj, dataIn)
            
            if 0
                %% OLD (disabled, to be abandoned in the future)
                nChan	= obj.sys.nChArius;
                nSamp	= obj.seq.nSamp;
                nTx     = obj.seq.nTx;
                nRep	= obj.seq.nRep;

                if obj.seq.hwDdcEnable
                    dataIn = reshape(dataIn, nChan, 2, nSamp, sum(obj.buffer.framesNumber));
                    dataIn = complex(dataIn(:,1,:,:), dataIn(:,2,:,:));
                end
                dataIn = reshape(dataIn, nChan, nSamp, sum(obj.buffer.framesNumber));
                dataIn = permute(dataIn, [2 1 3]);
                
                dataOut  = zeros(nSamp, obj.seq.rxApSize*nTx*nRep,'like',dataIn);
                dataOut(:,obj.buffer.reorgAddrDest) = dataIn(:, obj.buffer.reorgAddrOrig);
                dataOut  = reshape(dataOut, nSamp, obj.seq.rxApSize, nTx, nRep);
            else
                %% NEW
                dataIn  = gpuArray(dataIn);
                
                dataOut = rawReorg(dataIn, ...
                                   obj.buffer.reorgMap, ...
                                   uint32(obj.seq.rxApSize), ...
                                   uint32(obj.seq.nTx), ...
                                   uint32(obj.seq.nRep), ...
                                   obj.seq.hwDdcEnable);
            end
        end
        
    end
end
