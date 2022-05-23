classdef Us4R < handle
    % A handle to the Us4R system. 
    %
    % This class provides functions to configure the system and perform
    % data acquisition using the Us4R.
    % 
    % Available probe names:
    % 
    % - Esaote: 'AL2442', 'SL1543', 'AC2541', 'SP2430'
    % - Ultrasonix: 'L14-5/38'.
    % - ATL: 'L7-4'
    % 
    % Available adapter types:
    %
    % - 'esaote': 1st esaote adapter (no alternation in element handling),
    % - 'esaote2': 2nd esaote adapter (troublesome mapping),
    % - 'esaote3': current version of the esaote adapter,
    % - 'ultrasonix': current version of the ultrasonix probe adapter. 
    % - 'atl/philips': current version of the ATL/PHILIPS probe adapter.
    % 
    % Only one of the following parameters should be provided:
    %
    % :param voltage: a voltage to set, should be in range 0-90 [0.5*Vpp]. Required.
    % :param logTime: set to true if you want to display acquisition and reconstruction time. Optional.
    % :param probeName: name of the probe to use. The parameter is required when ``probe`` is not provided.
    % :param adapterType: name of the adapter type to use. The parameter is required when ``probe`` is not provided.
    % :param probe: definition of the probe to use. The parameter is required when ``probeName`` and ``adapterType`` are not provided.

    properties(Access = private)
        sys
        seq
        rec
        logTime
    end
    
    methods

        function obj = Us4R(varargin)
            [voltage, probeName, adapterType, interfaceName, logTime, probe] = Us4R.parseUs4RParams(varargin{:});

            obj.logTime = logTime;
            % System parameters
            obj.sys.nChArius = 32;
            
            obj.sys.trigTxDel = 240; % [samp] trigger to t0 (tx start) delay

            obj.sys.voltage = voltage;
            
            if(isempty(probe))
                probe = probeParams(probeName,adapterType,interfaceName);
            end
            
            obj.sys.nArius = probe.nUs4OEM; % number of Arius modules
            nArius = obj.sys.nArius;
            obj.sys.systemType = probe.systemType;
            
            % checking if voltage is safe
            isProperVoltageValue = @(x) ...
                   isnumeric(x) ...
                && isscalar(x) ...
                && isfinite(x) ...
                && x >= 0;

            if ~isProperVoltageValue(voltage)
                error('Invalid exctitation voltage value.')
            end
            
            if 2*voltage > probe.maxVpp
                error(['The electrical excitation exceeds the safe limit. ', ...
                       'For the current probe the limit is ', ...
                       num2str(probe.maxVpp/2), '[V].' ...
                       ])
            end
            
            
            obj.sys.adapType = probe.adapType;                      
            obj.sys.txChannelMap = probe.txChannelMap;
            obj.sys.rxChannelMap = probe.rxChannelMap;
            obj.sys.curvRadius = probe.curvRadius;
            obj.sys.probeMap = probe.probeMap;
            obj.sys.pitch = probe.pitch;
            obj.sys.nElem = probe.nElem;
            obj.sys.interfEnable = probe.interfEnable;

             % position (pos,x,z) and orientation (ang) of each probe element
             obj.sys.posElem = (-(probe.nElem-1)/2 : (probe.nElem-1)/2) * probe.pitch; % [m] (1 x nElem) position of probe elements along the probes surface
             if isnan(probe.curvRadius)
                 obj.sys.angElem = zeros(1,probe.nElem); % [rad] (1 x nElem) orientation of probe elements
                 obj.sys.xElem = obj.sys.posElem; % [m] (1 x nElem) z-position of probe elements
                 obj.sys.zElem = zeros(1,probe.nElem);% [m] (1 x nElem) x-position of probe elements
             else
                 obj.sys.angElem = obj.sys.posElem / -probe.curvRadius;
                 obj.sys.xElem = -probe.curvRadius * sin(obj.sys.angElem);
                 obj.sys.zElem = -probe.curvRadius * cos(obj.sys.angElem);
                 obj.sys.zElem = obj.sys.zElem - min(obj.sys.zElem);
             end
             
             if obj.sys.interfEnable
                 obj.sys.interfSize = probe.interfSize;
                 obj.sys.interfAng = probe.interfAng;
                 obj.sys.interfSos = probe.interfSos;
                 
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
             obj.sys.probeChannelsMask = probe.channelsMask;
%             obj.sys.maxVpp = probe.maxVpp;

            if obj.sys.adapType == 0
                % old adapter type (00001111)
                obj.sys.nChCont = obj.sys.nChArius;
                obj.sys.nChTotal = obj.sys.nChArius * 4 * nArius;
                
                obj.sys.selElem = (1:128).' + (0:(nArius-1))*128;
                obj.sys.actChan = true(128,nArius);
            elseif obj.sys.adapType == 1 || obj.sys.adapType == -1
                % place for new adapter support
                obj.sys.nChCont = obj.sys.nChArius * nArius;
                obj.sys.nChTotal = obj.sys.nChArius * 4 * nArius/abs(obj.sys.adapType);
                
                obj.sys.selElem = reshape(  (1:obj.sys.nChArius).' ...
                                          + (0:3)*obj.sys.nChArius*nArius, [], 1) ...
                                          + (0:(nArius-1))*obj.sys.nChArius;  % [nchannels x nArius]
                obj.sys.actChan = [true(96,nArius); false(32,nArius)];
                
            elseif obj.sys.adapType == 2
                % new adapter type (01010101)
                obj.sys.nChCont = obj.sys.nChArius * nArius;
                obj.sys.nChTotal = obj.sys.nChArius * 4 * nArius/obj.sys.adapType;
                
%                 obj.sys.selElem = reshape((1:obj.sys.nChArius).' + (0:3)*obj.sys.nChArius*nArius,[],1) + (0:(nArius-1))*obj.sys.nChArius;
%                 nChanTot = obj.sys.nChArius*4*nArius;
                obj.sys.selElem = repmat((1:128).',[1 nArius]);
                obj.sys.actChan = mod(ceil((1:128)' / obj.sys.nChArius) - 1, nArius) == (0:(nArius-1));
            elseif obj.sys.adapType == 3
                obj.sys.nChCont = obj.sys.nChArius * nArius;
                obj.sys.nChTotal = obj.sys.nChArius * 4 * nArius;
                
                obj.sys.selElem = reshape((1:obj.sys.nChArius).' + (0:3)*obj.sys.nChArius*nArius, [], 1) ...
                                + [(0:(nArius/2-1))*2, (0:(nArius/2-1))*2+1]*obj.sys.nChArius;  % [nchannels x nArius]
                
                obj.sys.actChan = [true(32, nArius); false(96, nArius)]; % [nchannels x nArius]
            else
                error("ARRUS:IllegalArgument", ['Unrecognized adapter type: ', obj.sys.adapType]);
            end
            obj.sys.actChan = obj.sys.actChan & any(obj.sys.selElem == reshape(obj.sys.probeMap, 1, 1, []),3);

        end

        function upload(obj, sequenceOperation, reconstructOperation)
            % Uploads operations to the us4R system.
            %
            % Currently, only supports :class:`SimpleTxRxSequence`
            % and :class:`Reconstruction` implementations.
            %
            % :param sequenceOperation: TX/RX sequence to perform on the us4R system
            % :param reconstructOperation: reconstruction to perform with the collected data
            % :returns: updated Us4R object
            
            switch(class(sequenceOperation))
                case 'PWISequence'
                    sequenceType = "pwi";
                case 'STASequence'
                    sequenceType = "sta";
                case "LINSequence"
                    sequenceType = "lin";
                otherwise
                    error("ARRUS:IllegalArgument", ...
                        ['Unrecognized operation type ', class(sequenceOperation)])
            end
            
            obj.setSeqParams(...
                'sequenceType', sequenceType, ...
                'txCenterElement', sequenceOperation.txCenterElement, ...
                'txApertureCenter', sequenceOperation.txApertureCenter, ...
                'txApertureSize', sequenceOperation.txApertureSize, ...
                'rxCenterElement', sequenceOperation.rxCenterElement, ...
                'rxApertureCenter', sequenceOperation.rxApertureCenter, ...
                'rxApertureSize', sequenceOperation.rxApertureSize, ...
                'txFocus', sequenceOperation.txFocus, ...
                'txAngle', sequenceOperation.txAngle, ...
                'speedOfSound', sequenceOperation.speedOfSound, ...
                'txFrequency', sequenceOperation.txFrequency, ...
                'txNPeriods', sequenceOperation.txNPeriods, ...
                'rxDepthRange', sequenceOperation.rxDepthRange, ...
                'rxNSamples', sequenceOperation.rxNSamples, ...
                'nRepetitions', sequenceOperation.nRepetitions, ...
                'txPri', sequenceOperation.txPri, ...
                'tgcStart', sequenceOperation.tgcStart, ...
                'tgcSlope', sequenceOperation.tgcSlope, ...
                'fsDivider', sequenceOperation.fsDivider, ...
                'txInvert', sequenceOperation.txInvert);
            
            % Validate compatibility of the sequence & the hardware
            obj.validateSequence;
            
            % Program hardware
            obj.programHW;
            
            if nargin==2
                obj.rec.enable = false;
                return;
            end
                
            obj.setRecParams(...
                'filterEnable', reconstructOperation.filterEnable, ...
                'filterACoeff', reconstructOperation.filterACoeff, ...
                'filterBCoeff', reconstructOperation.filterBCoeff, ...
                'filterDelay', reconstructOperation.filterDelay, ...
                'iqEnable', reconstructOperation.iqEnable, ...
                'cicOrder', reconstructOperation.cicOrder, ...
                'decimation', reconstructOperation.decimation, ...
                'xGrid', reconstructOperation.xGrid, ...
                'zGrid', reconstructOperation.zGrid, ...
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
            % Currently, only supports :class:`SimpleTxRxSequence` and :class:`Reconstruction`
            % implementations.
            %
            % :returns: RF frame and reconstructed image (if :class:`Reconstruction` operation was uploaded)
            
            obj.openSequence;
            [rf, ~] = obj.execSequence;
            obj.closeSequence;
            
            if obj.rec.enable
                img = obj.execReconstr(rf(:,:,:,1));
            else
                img = [];
            end
        end
        
        function [rf, img, metadata] = runWithMetadata(obj)
            % Runs uploaded operations in the us4R system.
            %
            % Currently, only supports :class:`SimpleTxRxSequence` and :class:`Reconstruction`
            % implementations.
            %
            % :returns: RF frame, reconstructed image (if :class:`Reconstruction` operation was uploaded) and metadata located in the first sample of the master module
            obj.openSequence;
            [rf, metadata] = obj.execSequence;
            obj.closeSequence;
            
            if obj.rec.enable
                img = obj.execReconstr(rf(:,:,:,1));
            else
                img = [];
            end
        end
        
        function runLoop(obj, isContinue, callback)
            % Runs the uploaded operations in a loop.
            % 
            % Currently, only supports :class:`SimpleTxRxSequence` and \
            % :class:`Reconstruction` implementations.
            %
            % :param isContinue: should the system continue executing \
            %   the op? Takes no parameters and returns a boolean value.
            % :param callback: a function to call after executing the \
            %   operation. Should take one parameter, which will be feed with \
            %   the output of the executed op.
            
            obj.openSequence;
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
            obj.closeSequence;
        end
        
        function prepareBuffer(obj, nElements)
            Us4MEX(0, "PrepareInternalBuffer", nElements, obj.seq.nSamp*obj.seq.nTrig);
        end
        
        function acquireToBuffer(obj, nElements)
            obj.openSequence;
            Us4MEX(0, "AcquireToInternalBuffer", nElements, obj.seq.txPri*obj.seq.nTrig);
            obj.closeSequence;
        end
        
        function [rf] = popBufferElement(obj)
            rf = Us4MEX(0, "PopInternalBufferElement");
        end
        
        function [img] = reconstructOffline(obj,rfRaw)
            img = obj.execReconstr(rfRaw);
        end
    end
    
    methods(Access = private, Static)
        
        function [voltage, probeName, adapterType, interfaceName, logTime, probe] = parseUs4RParams(varargin)
           paramsParser = inputParser;
           addParameter(paramsParser, 'nUs4OEM', []);
           addParameter(paramsParser, 'voltage', []);
           addParameter(paramsParser, 'logTime', false);
           addParameter(paramsParser, 'probeName', []);
           addParameter(paramsParser, 'adapterType', []);
           addParameter(paramsParser, 'interfaceName', 'none');
           addParameter(paramsParser, 'probe', []);
           parse(paramsParser, varargin{:});

           nArius = paramsParser.Results.nUs4OEM;
           % TODO remove the nUs4OEM parameter value
           if(~isempty(nArius))
               warning("Parameter 'nUs4OEM' is deprecated and will be ignored.");
           end
           voltage = paramsParser.Results.voltage;
           if(~isscalar(voltage))
               error("ARRUS:IllegalArgument", ...
               "Parameter voltage is required and should be a scalar");
           end
           logTime = paramsParser.Results.logTime;

           % First option
           probeName = paramsParser.Results.probeName;
           adapterType = paramsParser.Results.adapterType;
           interfaceName = paramsParser.Results.interfaceName;
           if xor(isempty(probeName), isempty(adapterType))
               error("ARRUS:IllegalArgument", ...
               "All or none of the following parameters are required: probeName, adapterType");
           end
           % Consider reorganizing the 1st and 2nd option conditions

           % Second option
           probe = paramsParser.Results.probe;
           if ~xor(isempty(probe), isempty(probeName))
               error("ARRUS:IllegalArgument", ...
                 "Exactly one of the following parameter should be provided: probe, pair(probeName, adapterType)");
           end
       end
    end

    methods(Access = private)

        function setSeqParams(obj,varargin)

            %% Set sequence parameters
            % Sequence parameters names mapping
            %                    public name         private name
            seqParamMapping = { 'sequenceType',     'type'
                                % aperture
                                'txCenterElement',  'txCentElem'; ...
                                'txApertureCenter', 'txApCent'; ...
                                'txApertureSize',   'txApSize'; ...
                                'rxCenterElement',  'rxCentElem'; ...
                                'rxApertureCenter', 'rxApCent'; ...
                                'rxApertureSize',   'rxApSize'; ...
                                'txFocus',          'txFoc'; ...
                                'txAngle',          'txAng'; ...
                                'speedOfSound',     'c'; ...
                                'txFrequency',      'txFreq'; ...
                                'txNPeriods',       'txNPer'; ...
                                'rxDepthRange',     'dRange'; ...
                                'rxNSamples',       'nSamp'; ...
                                'nRepetitions',     'nRep'; ...
                                'txPri',            'txPri'; ...
                                'tgcStart',         'tgcStart'; ...
                                'tgcSlope',         'tgcSlope'; ...
                                'fsDivider'         'fsDivider'; ...
                                'txInvert'          'txInvert'};

            for iPar=1:size(seqParamMapping,1)
                obj.seq.(seqParamMapping{iPar,2}) = [];
            end

            nPar = length(varargin)/2;
            for iPar=1:nPar
                idPar = strcmpi(varargin{iPar*2-1},seqParamMapping(:,1));
                obj.seq.(seqParamMapping{idPar,2}) = reshape(varargin{iPar*2},1,[]);
            end
            
            %% Fixed parameters
            obj.seq.rxSampFreq	= 65e6./obj.seq.fsDivider; % [Hz] sampling frequency
            obj.seq.rxDel       = 0e-6;
            obj.seq.rxTime      = obj.seq.txPri - obj.seq.rxDel - 5e-6;	% [s] rx time (max 4000us)
            obj.seq.pauseMultip	= 1.5;
            
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
            
            %% Default txPri
            if isempty(obj.seq.txPri)
                obj.seq.txPri = (obj.seq.startSample + obj.seq.nSamp) / obj.seq.rxSampFreq + 40e-6;
            end
            
            %% TGC
            % Default TGC
            if isempty(obj.seq.tgcStart)
                obj.seq.tgcStart = 14;
            end
            if isempty(obj.seq.tgcSlope)
                obj.seq.tgcSlope = 2 * 0.5e2 * obj.seq.txFreq*1e-6;
            end
            
            distance = (round(400/obj.seq.fsDivider) : ...
                        round(150/obj.seq.fsDivider) : ...
                        (obj.seq.startSample + obj.seq.nSamp - 1)) / obj.seq.rxSampFreq * obj.seq.c;         % [m]
            tgcCurve = obj.seq.tgcStart + obj.seq.tgcSlope * distance;  % [dB]
            if any(tgcCurve<14 | tgcCurve>54)
                warning('TGC values are limited to 14-54dB range');
                tgcCurve = max(14,min(54,tgcCurve));
            end
            
            tgcChar = [14.000, 14.001, 14.002, 14.003, 14.024, 14.168, 14.480, 14.825, 15.234, 15.770, ...
                       16.508, 17.382, 18.469, 19.796, 20.933, 21.862, 22.891, 24.099, 25.543, 26.596, ...
                       27.651, 28.837, 30.265, 31.690, 32.843, 34.045, 35.543, 37.184, 38.460, 39.680, ...
                       41.083, 42.740, 44.269, 45.540, 46.936, 48.474, 49.895, 50.966, 52.083, 53.256, 54];
            tgcCurve = interp1(tgcChar,14:54,tgcCurve);
            
            obj.seq.tgcCurve = (tgcCurve-14) / 40;                      % <0,1>
            
            %% Tx/Rx aperture string/missing parameters
            if isstring(obj.seq.txApSize) && obj.seq.txApSize == "nElements"
                obj.seq.txApSize = min(obj.sys.nChTotal,obj.sys.nElem);
                disp(['txApertureSize set to ' num2str(obj.seq.txApSize) '.']);
            end
            
            if isstring(obj.seq.rxApSize)
                if obj.seq.rxApSize == "nChannels"
                    obj.seq.rxApSize = obj.sys.nChCont;
                elseif obj.seq.rxApSize == "nElements"
                    obj.seq.rxApSize = min(obj.sys.nChTotal,obj.sys.nElem);
                end
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
            if obj.sys.interfEnable && (~strcmp(obj.seq.type,"sta") || obj.seq.txApSize~=1)
                error("setSeqParams: only SSTA scheme is supported when wedge interface is used");
            end
            
            %% Aperture masks & delays
            obj.calcTxRxApMask;
            obj.calcTxDelays;
            
            %% Number of: Tx, SubTx, Firings, Triggers
            obj.seq.nTx	= length(obj.seq.txAng);
            
            nSubTx = zeros(1,obj.sys.nArius);
            for iArius=0:(obj.sys.nArius-1)
                rxApMaskSelect = obj.seq.rxApMask(obj.sys.selElem(:,iArius+1), :) & obj.sys.actChan(:,iArius+1);
                iSubTx = cumsum(reshape(rxApMaskSelect,obj.sys.nChArius,4,obj.seq.nTx),2);
                nSubTx(iArius+1) = max(iSubTx(:));
            end
            obj.seq.nSubTx = max(nSubTx);
            obj.seq.nFire = obj.seq.nTx * obj.seq.nSubTx;
            
            if isstring(obj.seq.nRep) && obj.seq.nRep == "max"
                obj.seq.nRep = min(floor([ ...
                                2^14 / obj.seq.nFire, ...
                                2^32 / obj.seq.nFire / (obj.sys.nChArius * obj.seq.nSamp * 2)]));
                disp(['nRepetitions set to ' num2str(obj.seq.nRep) '.']);
            end
            obj.seq.nTrig = obj.seq.nFire * obj.seq.nRep;
            
            %% Sub-sequence parameters
            obj.calcTxRxSubParams;
            
            %% Active channels group mask
            if obj.sys.adapType == -1
                [~,I] = sort(obj.sys.rxChannelMap.');
                obj.seq.actChanGroupMask = reshape(any(reshape(obj.sys.actChan(I + (0:(nArius-1))*128), 8, 16, [])), 16, []);
                % for future: some other adapters (esaote, atl/philips)
                % can have a similar problem as esaote2 but on a much smaller scale
            else
                obj.seq.actChanGroupMask = reshape(any(reshape(obj.sys.actChan, 8, 16, [])), 16, []);
            end

        end

        function setRecParams(obj,varargin)
            %% Set reconstruction parameters
            % Reconstruction parameters names mapping
            %                    public name         private name
            recParamMapping = { 'filterEnable',     'filtEnable'; ...
                                'filterACoeff',     'filtA'; ...
                                'filterBCoeff',     'filtB'; ...
                                'filterDelay',      'filtDel'; ...
                                'iqEnable',         'iqEnable'; ...
                                'cicOrder',         'cicOrd'; ...
                                'decimation',       'dec'; ...
                                'xGrid',            'xGrid'; ...
                                'zGrid',            'zGrid'; ...
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
            
            %% Default decimation
            if isempty(obj.rec.dec)
                obj.rec.dec = round(obj.seq.rxSampFreq / obj.seq.txFreq);
            end
            
            %% Default bmodeFrames
            if obj.rec.bmodeEnable && isempty(obj.rec.bmodeFrames)
                obj.rec.bmodeFrames = 1:obj.seq.nTx;
            end

            %% Resulting parameters
            obj.rec.zSize	= length(obj.rec.zGrid);
            obj.rec.xSize	= length(obj.rec.xGrid);
            
            if obj.rec.colorEnable || obj.rec.vectorEnable
                [~,obj.rec.wcFiltInitCoeff] = filter(obj.rec.wcFiltB,obj.rec.wcFiltA,ones(1000,1));
            end
            
            %% If GPU is available...
            obj.rec.gpuEnable	= license('test', 'Distrib_Computing_Toolbox') && ~isempty(ver('distcomp')) && parallel.gpu.GPUDevice.isAvailable;
            
            if obj.rec.gpuEnable && ~strcmp(obj.seq.type,'lin')
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
                obj.seq.txApFstElem    = gpuArray( int32(obj.seq.txApFstElem - 1));
                obj.seq.txApLstElem    = gpuArray( int32(obj.seq.txApLstElem - 1));
                obj.seq.rxApFstElem    = gpuArray( int32(obj.seq.rxApOrig - 1));    % rxApOrig remains unchanged as it is used in data reorganization
                obj.seq.nSampOmit      = gpuArray( int32(((max(obj.seq.txDel) + obj.seq.txNPer/obj.seq.txFreq) * obj.seq.rxSampFreq + 50) / obj.rec.dec));
                obj.rec.bmodeRxTangLim =          single(obj.rec.bmodeRxTangLim);
                obj.rec.colorRxTangLim =          single(obj.rec.colorRxTangLim);
                obj.rec.vect0RxTangLim =          single(obj.rec.vect0RxTangLim);
                obj.rec.vect1RxTangLim =          single(obj.rec.vect1RxTangLim);
                obj.seq.rxSampFreq     =          single(obj.seq.rxSampFreq);
                obj.seq.txFreq         =          single(obj.seq.txFreq);
                obj.rec.dec            =          single(obj.rec.dec);
                obj.seq.c              =          single(obj.seq.c);
                obj.seq.startSample    =          single(obj.seq.startSample);
                obj.seq.txDelCent      =          single(obj.seq.txDelCent);
                obj.seq.txNPer         =          single(obj.seq.txNPer);
                
            end
        end
        
        function val = get(obj,paramName)

            if isfield(obj.sys,paramName)
                val = obj.sys.(paramName);
            else
                if isfield(obj.seq,paramName)
                    val = obj.seq.(paramName);
                else
                    if isfield(obj.rec,paramName)
                        val = obj.rec.(paramName);
                    else
                        error('Invalid parameter name');
                    end
                end
            end

        end
        
        function calcTxRxApMask(obj)
            % calcTxRxApMask appends the following fields to the in/out obj:
            % obj.seq.txApOrig      - [element] (1 x nTx) number of probe element being the origin of the tx aperture
            % obj.seq.rxApOrig      - [element] (1 x nTx) number of probe element being the origin of the rx aperture
            % obj.seq.txApFstElem	- [element] (1 x nTx) number of probe element being the first in the tx aperture
            % obj.seq.txApLstElem	- [element] (1 x nTx) number of probe element being the last in the tx aperture
            % obj.seq.txApMask      - [logical] (nChTotal x nTx) tx aperture mask
            % obj.seq.rxApMask      - [logical] (nChTotal x nTx) rx aperture mask
            % obj.seq.rxElemId      - [element] (nChTotal x nTx) element numbering in the rx aperture
            
            iElem = nan(1,max(obj.sys.probeMap));
            iElem(obj.sys.probeMap) = 1:obj.sys.nElem;
            if length(iElem) >= obj.sys.nChTotal
                iElem = iElem(1:obj.sys.nChTotal);
            else
                iElem = [iElem, nan(1, obj.sys.nChTotal-length(iElem))];
            end
            

            obj.seq.txApOrig = round(obj.seq.txCentElem - (obj.seq.txApSize-1)/2 + 1e-9);
            obj.seq.rxApOrig = round(obj.seq.rxCentElem - (obj.seq.rxApSize-1)/2 + 1e-9);
            
            obj.seq.txApFstElem = max(1,          obj.seq.txApOrig);
            obj.seq.txApLstElem = min(max(iElem), obj.seq.txApOrig + obj.seq.txApSize - 1);
            
            obj.seq.txApMask = (iElem.' >= obj.seq.txApOrig) & (iElem.' <= obj.seq.txApOrig + obj.seq.txApSize - 1);
            systemChannelsMask = true(size(obj.seq.txApMask));
            systemChannelsMask(obj.sys.probeMap(obj.sys.probeChannelsMask), :) = false;
            obj.seq.txApMask = obj.seq.txApMask & systemChannelsMask;
            obj.seq.rxApMask = (iElem.' >= obj.seq.rxApOrig) & (iElem.' <= obj.seq.rxApOrig + obj.seq.rxApSize - 1);
            obj.seq.rxApMask = obj.seq.rxApMask & systemChannelsMask;
            obj.seq.rxElemId = (iElem.' - obj.seq.rxApOrig + 1) .* obj.seq.rxApMask;
            obj.seq.rxElemId(isnan(obj.seq.rxElemId)) = 0;
        end

        function calcTxDelays(obj)
            % calcTxDelays appends the following fields to the in/out obj:
            % obj.seq.txDel         - [s] (nArius*128 x nTx) tx delays for each element
            % obj.seq.txDelCent     - [s] (1 x nTx) tx delays for tx aperture centers
            
            nElem = max(obj.sys.probeMap);
            xElem = nan(1,nElem);
            zElem = nan(1,nElem);
            xElem(obj.sys.probeMap) = obj.sys.xElem;
            zElem(obj.sys.probeMap) = obj.sys.zElem;
            if nElem >= obj.sys.nChTotal
                xElem = xElem(1:obj.sys.nChTotal);
                zElem = zElem(1:obj.sys.nChTotal);
            else
                xElem = [xElem, nan(1, obj.sys.nChTotal-nElem)];
                zElem = [zElem, nan(1, obj.sys.nChTotal-nElem)];
            end
            
            %% CALCULATE DELAYS
            sos = obj.seq.c;
            
            if isinf(obj.seq.txFoc)
                % Delays due to the tilting the plane wavefront
                txDel       = (xElem.'           .* sin(obj.seq.txAngZX) + ...
                               zElem.'           .* cos(obj.seq.txAngZX)) / sos;            % [s] (nElem x nTx) delays for tx elements
                txDelCent	= (obj.seq.txApCentX .* sin(obj.seq.txAngZX) + ...
                               obj.seq.txApCentZ .* cos(obj.seq.txAngZX)) / sos;            % [s] (1 x nTx) delays for tx aperture center
            else
                % Focal point positions
                xFoc        = obj.seq.txApCentX + obj.seq.txFoc .* sin(obj.seq.txAngZX);	% [m] (1 x nTx) x-position of the focal point
                zFoc        = obj.seq.txApCentZ + obj.seq.txFoc .* cos(obj.seq.txAngZX);	% [m] (1 x nTx) z-position of the focal point
                
                % Delays due to the element - focal point distances
                nTx         = numel(xFoc);
                txDel       = nan(obj.sys.nChTotal,nTx);
                txDelCent	= nan(1,nTx);
                for iTx=1:nTx
                    txDel(:,iTx)	= sqrt( (xFoc(iTx) -         xElem.'       ).^2 + ...
                                            (zFoc(iTx) -         zElem.'       ).^2) / sos;	% [s] (nElem x nTx) delays for tx elements
                    txDelCent(iTx)	= sqrt( (xFoc(iTx) - obj.seq.txApCentX(iTx)).^2 + ...
                                            (zFoc(iTx) - obj.seq.txApCentZ(iTx)).^2) / sos;	% [s] (1 x nTx) delays for tx aperture center
                end
                
                % Inverse the delays for the 'focusing' option (txFoc>0)
                % For 'defocusing' the delays remain unchanged
                focDefoc	= 1 - 2 * double(obj.seq.txFoc>0);
                txDel       = txDel     .* focDefoc;
                txDelCent	= txDelCent .* focDefoc;
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
        
        function calcTxRxSubParams(obj)
            nArius	= obj.sys.nArius;
            nChan	= obj.sys.nChArius;
            nSubTx	= obj.seq.nSubTx;
            nTx     = obj.seq.nTx;
            nFire	= obj.seq.nFire;
            
            txSubApDel = cell(nArius,nTx);
            txSubApMask = false(128,nTx,nArius);
            rxSubApMask = false(128,nFire,nArius);
            rxSubElemId = zeros(128,nFire,nArius);
            for iArius=0:(nArius-1)
                txSubApDel(iArius+1,:) = mat2cell(obj.seq.txDel(obj.sys.selElem(:,iArius+1), :) .* obj.sys.actChan(:,iArius+1), 128, ones(1,nTx));
                txSubApMask(:,:,iArius+1) = obj.seq.txApMask(obj.sys.selElem(:,iArius+1), :) & obj.sys.actChan(:,iArius+1);
                
                rxApMaskSelect = obj.seq.rxApMask(obj.sys.selElem(:,iArius+1), :) & obj.sys.actChan(:,iArius+1);
                rxApMaskSelect = reshape(rxApMaskSelect,nChan,4,nTx);
                iSubTx = cumsum(rxApMaskSelect,2) .* rxApMaskSelect;
                iSubTx = reshape(iSubTx,[],1,nTx);
                rxSubApMask(:,:,iArius+1) = reshape(iSubTx == (1:nSubTx),[],nFire);
                
                % rxSubApMask correction for the new esaote adapter
                if obj.sys.adapType == -1
                    for iFire=0:(nFire-1)
                        rxSubChanMap = 1+mod(obj.sys.rxChannelMap(iArius+1,rxSubApMask(:,iFire+1,iArius+1))-1,obj.sys.nChArius);
                        rejElem = floor((find(triu(rxSubChanMap==rxSubChanMap.',1)) - 1) / length(rxSubChanMap)) + 1;
                        if ~isempty(rejElem)
                            elemIdx = cumsum(rxSubApMask(:,iFire+1,iArius+1)) .* rxSubApMask(:,iFire+1,iArius+1);
                            rejElem = any(elemIdx == rejElem.', 2);
                            rxSubApMask(rejElem,iFire+1,iArius+1) = false;
                        end
                    end
                end
                
                rxElemIdSelect = obj.seq.rxElemId(obj.sys.selElem(:,iArius+1), :) .* obj.sys.actChan(:,iArius+1);
                rxSubElemId(:,:,iArius+1) = reshape(reshape(rxElemIdSelect,[],1,nTx) .* ...
                                                    reshape(rxSubApMask(:,:,iArius+1),[],nSubTx,nTx), [],nFire);
            end
            
            obj.seq.txSubApMask = txSubApMask;
            obj.seq.txSubApDel = txSubApDel;
            obj.seq.rxSubApMask = rxSubApMask;
            obj.seq.rxSubElemId = rxSubElemId;
        end
        
        function validateSequence(obj)
            
            %% Validate number of firings
            if obj.seq.nFire > 2048
                error("ARRUS:IllegalArgument", ...
                        ['Number of firings (' num2str(obj.seq.nFire) ') cannot exceed 1024.' ]);
            end
            
            %% Validate number of triggers
            if obj.seq.nTrig > 16384
                error("ARRUS:IllegalArgument", ...
                        ['Number of triggers (' num2str(obj.seq.nTrig) ') cannot exceed 16384.']);
            end
            
            %% Validate number of samples
            if obj.seq.nSamp > 65536/obj.seq.fsDivider
                error("ARRUS:IllegalArgument", ...
                        ['Number of samples ' num2str(obj.seq.nSamp) ' cannot exceed ' num2str(65536/obj.seq.fsDivider) '.'])
            end
            
            if mod(obj.seq.nSamp,64) ~= 0
                error("ARRUS:IllegalArgument", ...
                        ['Number of samples (' num2str(obj.seq.nSamp) ') must be divisible by 64.']);
            end
            
            %% Validate memory usage
            memoryRequired = obj.sys.nChArius * obj.seq.nSamp * 2 * obj.seq.nTrig;  % [B]
            if memoryRequired > 2^32  % 4GB
                error("ARRUS:OutOfMemory", ...
                        ['Required memory per module (' num2str(memoryRequired/2^30) 'GB) cannot exceed 4GB.']);
            end
            
        end
        
        function programHW(obj)
            
            % Unloading Us4MEX should clear the device state.
            munlock('Us4MEX');
            clear Us4MEX;
            
            %% Program mappings, gains, and voltage
            
            
            Us4MEX(0, "Initialize", obj.sys.systemType, obj.sys.nArius);
            try
                Us4MEX(0,"EnableHV");
            catch
                warning('1st "EnableHV" failed');
                Us4MEX(0,"EnableHV");
            end
                
            try
                Us4MEX(0, "SetHVVoltage", obj.sys.voltage);
            catch
                warning('1st "SetHVVoltage" failed');
                Us4MEX(0, "SetHVVoltage", obj.sys.voltage);
            end
            
            for iArius=0:(obj.sys.nArius-1)
                % Set Rx channel mapping
%                 for iChan=1:32
%                     Us4MEX(iArius, "SetRxChannelMapping", obj.sys.rxChannelMap(iArius+1,iChan), iChan);
%                 end
                for iFire=0:(obj.seq.nFire-1)
                    Us4MEX(iArius, "SetRxChannelMapping", obj.sys.rxChannelMap(iArius+1,1:32), iFire);
                end

                % Set Tx channel mapping
                for iChan=1:128
                    Us4MEX(iArius, "SetTxChannelMapping", obj.sys.txChannelMap(iArius+1,iChan), iChan);
                end

                % init RX
                Us4MEX(iArius, "SetPGAGain","30dB");
                Us4MEX(iArius, "SetLPFCutoff","15MHz");
                Us4MEX(iArius, "SetActiveTermination","EN", "200");
                Us4MEX(iArius, "SetLNAGain","24dB");
                Us4MEX(iArius, "SetDTGC","DIS", "0dB");                 % EN/DIS? (attenuation actually, 0:6:42)
                Us4MEX(iArius, "TGCEnable");

                
            end
            
            %% Program Tx/Rx sequence
            for iArius=0:(obj.sys.nArius-1)
                Us4MEX(iArius, "SetNumberOfFirings", obj.seq.nFire);
                Us4MEX(iArius, "ClearScheduledReceive");

                for iFire=0:(obj.seq.nFire-1)
                    Us4MEX(iArius, "SetActiveChannelGroup", obj.maskFormat(obj.seq.actChanGroupMask(:,iArius+1)), iFire);
                    
                    %% Tx
                    iTx     = 1 + floor(iFire/obj.seq.nSubTx);
                    Us4MEX(iArius, "SetTxAperture", obj.maskFormat(obj.seq.txSubApMask(:,iTx,iArius+1)), iFire);
                    Us4MEX(iArius, "SetTxDelays", obj.seq.txSubApDel{iArius+1,iTx}, iFire);
                    Us4MEX(iArius, "SetTxFrequency", obj.seq.txFreq, iFire);
                    Us4MEX(iArius, "SetTxHalfPeriods", obj.seq.txNPer*2, iFire);
                    Us4MEX(iArius, "SetTxInvert", obj.seq.txInvert, iFire);
                    
                    %% Rx
                    % SetRxChannelMapping for the new esaote adapter
                    if obj.sys.adapType == -1
                        rxSubChanIdx = find(obj.seq.rxSubApMask(:,iFire+1,iArius+1));
                        rxSubChanMap = obj.sys.rxChannelMap(iArius+1,rxSubChanIdx);
                        rxSubChanIdx = 1+mod(rxSubChanIdx-1,obj.sys.nChArius);
                        rxSubChanMap = 1+mod(rxSubChanMap-1,obj.sys.nChArius);
                        rxMapping = nan(1, 32);
                        rxMapping(rxSubChanIdx) = rxSubChanMap;
                        % Mapping inactive elements to inactive channels
                        % They are ommited in execSequence anyway
                        rxMapping(isnan(rxMapping)) = find(~any((1:obj.sys.nChArius).' == rxSubChanMap, 2));
                        
                        Us4MEX(iArius, "SetRxChannelMapping", rxMapping, iFire);
                    end
                    Us4MEX(iArius, "SetRxAperture", obj.maskFormat(obj.seq.rxSubApMask(:,iFire+1,iArius+1)), iFire);
                    Us4MEX(iArius, "SetRxTime", obj.seq.rxTime, iFire);
                    Us4MEX(iArius, "SetRxDelay", obj.seq.rxDel, iFire);
                    Us4MEX(iArius, "TGCSetSamples", obj.seq.tgcCurve, iFire);
                    
                end

                Us4MEX(iArius, "EnableTransmit");
            end
            
             %% Program recording
            % ScheduleReceive should be done for every iTrig (not iFire)
            % This loop will be probably used in the future
            for iArius=0:(obj.sys.nArius-1)
                for iTrig=0:(obj.seq.nTrig-1)
                    iFire = mod(iTrig, obj.seq.nFire);
                    Us4MEX(iArius, "ScheduleReceive", iTrig, iTrig*obj.seq.nSamp, ...
                                    obj.seq.nSamp, ... 
                                    obj.seq.startSample + obj.sys.trigTxDel, ...
                                    obj.seq.fsDivider-1, iFire);
                end
            end
            
            %% Program triggering
            for iArius=0:(obj.sys.nArius-1)
                Us4MEX(iArius, "SetNTriggers", obj.seq.nTrig);
                for iTrig=0:(obj.seq.nTrig-1)
                    Us4MEX(iArius, "SetTrigger", obj.seq.txPri*1e6, 0, iTrig);
                end
                Us4MEX(iArius, "SetTrigger", obj.seq.txPri*1e6, 1, obj.seq.nTrig-1);
                
                Us4MEX(iArius, "EnableSequencer");      % loading parameters after TX/RX
%                 Us4MEX(iArius, "EnableSequencer", 1);   % loading parameters during TX/RX
            end
            
           
        end

        function openSequence(obj)
            %% Start acquisitions (1st sequence exec., no transfer to host)
            Us4MEX(0, "TriggerStart");
            pause(obj.seq.pauseMultip * obj.seq.txPri * obj.seq.nTrig);
        end

        function closeSequence(obj)
            %% Stop acquisition
            Us4MEX(0, "TriggerStop");

        end

        function [rf, metadata] = execSequence(obj)

            nArius	= obj.sys.nArius;
            nChan	= obj.sys.nChArius;
            nSamp	= obj.seq.nSamp;
            nSubTx	= obj.seq.nSubTx;
            nTx     = obj.seq.nTx;
            nRep	= obj.seq.nRep;
            nTrig	= nTx*nSubTx*nRep;

            %% Capture data
            Us4MEX(0, "TriggerSync");
            pause(obj.seq.pauseMultip * obj.seq.txPri * nTrig);
            
            %% Transfer to PC
            rf = Us4MEX(0, ...
                        "TransferAllRXBuffersToHost",  ...
                        zeros(nArius, 1), ...
                        repmat(nSamp * nTrig, [nArius 1]), ...
                        int8(obj.logTime) ...
            );
            %% Get metadata
            metadata = zeros(nChan, nTrig, 'int16');
            metadata(:, :) = rf(:, 1:nSamp:nTrig*nSamp);
            %% Reorganize
            rf	= reshape(rf, [nChan, nSamp, nSubTx, nTx, nRep, nArius]);

            rxApOrig = obj.seq.rxApOrig;
            if obj.sys.adapType == 0
                rf	= permute(rf,[2 1 3 6 4 5]);
                
                % "old esaote" adapter type
%                 for iTx=1:nTx
%                     rf(:,:,iTx,:)	= circshift(rf(:,:,iTx,:),-min(32,max(0,rxApOrig(iTx)-1-nChan*(4-1))),2);
%                 end
%                 rf	= rf(:,1:nChan,:,:);
%                 for iTx=1:nTx
%                     if ~(rxApOrig(iTx) > 1+nChan*(4-1) && rxApOrig(iTx) < 1+nChan*4)
%                         rf(:,:,iTx,:)	= circshift(rf(:,:,iTx,:),-mod(rxApOrig(iTx)-1,nChan),2);
%                     end
%                 end
                
                for iTx=1:nTx
                    iArius = ceil(rxApOrig(iTx) / (nChan * 4)) - 1;
                    if iArius >= 0 && iArius < nArius
                        rf(:,:,:,iArius+1,iTx,:)	= circshift(rf(:,:,:,iArius+1,iTx,:),-mod(rxApOrig(iTx)-1,nChan),2);
                    end
                end
                rf = reshape(rf,nSamp,nChan*nSubTx*nArius,nTx,nRep);
                for iTx=1:nTx
                    rxApEnd = rxApOrig(iTx) + obj.seq.rxApSize - 1;
                    nZerosL = max(0, min(  0,rxApEnd) -          rxApOrig(iTx)  + 1);
                    nZerosR = max(0,         rxApEnd  - max(193, rxApOrig(iTx)) + 1);
                    nChan0  = max(0, min(128,rxApEnd) - max(  1, rxApOrig(iTx)) + 1);
                    nChan1  = max(0, min(192,rxApEnd) - max(129, rxApOrig(iTx)) + 1);
                    
                    rf(:,1:obj.seq.rxApSize,iTx,:) = [  zeros(nSamp,nZerosL,1,nRep), ...
                                                        rf(:,(1:nChan0) + 0*nChan*nSubTx,iTx,:), ...
                                                        rf(:,(1:nChan1) + 1*nChan*nSubTx,iTx,:), ...
                                                        zeros(nSamp,nZerosR,1,nRep) ];
                end
                rf(:,(obj.seq.rxApSize+1):end,:,:) = [];
                
            else
                % "ultrasonix" or "new esaote" adapter type
                rf0	= permute(rf,[2 1 3 4 6 5]);
                rf0	= reshape(rf0,nSamp,nChan*nSubTx*nTx*nArius,nRep);
                rf	= zeros(nSamp,obj.seq.rxApSize*nTx,nRep,'int16');
                
                addressFrom = reshape(any(reshape(obj.seq.rxSubApMask,nChan,4,[]),2),[],1);
                addressTo   = reshape(sum(reshape(obj.seq.rxSubElemId,nChan,4,[]),2),[],1);
                addressTo   = addressTo + reshape((0:(nTx-1)).*ones(nChan*nSubTx,1,nArius),[],1)*obj.seq.rxApSize;
                addressTo   = addressTo(addressFrom);
                
                rf(:,addressTo,:) = rf0(:,addressFrom,:);
                rf	= reshape(rf,nSamp,obj.seq.rxApSize,nTx,nRep);
                
            end

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
            rfRaw = downConversion(rfRaw,obj.seq,obj.rec);
            
            % warning: both filtration and decimation introduce phase delay!
            % rfRaw = preProc(rfRaw,obj.seq,obj.rec);

            %% Reconstruction
            if strcmp(obj.seq.type,'lin')
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
            if ~obj.rec.iqEnable
                nanMask = isnan(rfBfr);
                rfBfr(nanMask) = 0;
                rfBfr = hilbert(rfBfr);
                rfBfr(nanMask) = nan;
            end
            
            % Envelope detection
            envImg = abs(rfBfr);
            
            % Scan conversion (for 'lin' mode)
            if strcmp(obj.seq.type,'lin')
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

        function maskString = maskFormat(obj,maskLogical)
            
            [maskLength,nMask] = size(maskLogical);
            
            if maskLength~=16 && maskLength~=128
                error("maskFormat: invalid mask length, should be 16 or 128");
            end
            
            if maskLength == 16
                % active channel group mask: needs reordering
                maskLogical = reshape(permute(reshape(maskLogical,4,2,2,nMask),[3,2,1,4]),16,nMask);
            end
            
            maskString = join(string(double(maskLogical.')),"").';
            maskString = reverse(maskString);
            
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
                                    obj.seq.txApFstElem(selFrames), ...
                                    obj.seq.txApLstElem(selFrames), ...
                                    obj.seq.rxApFstElem(selFrames), ...
                                    obj.seq.nSampOmit(selFrames), ...
                                    rxTangLim(1), ...
                                    rxTangLim(2), ...
                                    obj.seq.rxSampFreq / obj.rec.dec, ...
                                    obj.seq.txFreq, ...
                                    obj.seq.c, ...
                                    - obj.seq.startSample/obj.seq.rxSampFreq + obj.seq.txDelCent + obj.seq.txNPer/(2*obj.seq.txFreq));
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
                                    obj.seq.rxApFstElem(selFrames), ...
                                    rxTangLim(1), ...
                                    rxTangLim(2), ...
                                    obj.seq.rxSampFreq / obj.rec.dec, ...
                                    obj.seq.txFreq, ...
                                    obj.seq.c, ...
                                    obj.sys.interfSos, ...
                                    1/64/obj.seq.txFreq, ...
                                    - obj.seq.startSample/obj.seq.rxSampFreq + obj.seq.txDelCent + obj.seq.txNPer/(2*obj.seq.txFreq));
            end
            
        end
        
    end
end
