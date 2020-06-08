classdef Us4R < handle
    % A handle to the Us4R system. 
    %
    % This class provides functions to configure the system and perform
    % data acquisition using the Us4R.
    %
    % :param nArius: number of Us4OEM modules available in the us4R system
    % :param probeName: name of the probe to use, available: \
    %    Esaote: 'AL2442', 'SL1543', \
    %    Ultrasonix: 'L14-5/38'
    % :param voltage: a voltage to set, should be in range 0-90 [0.5*Vpp]
    % :param logTime: set to true if you want to display acquisition \
    %    and reconstruction time (optional)

    properties(Access = private)
        sys
        seq
        rec
        logTime
    end
    
    methods

        function obj = Us4R(nArius, probeName, voltage, logTime)
            if nargin < 4
                obj.logTime = false;
            else
                obj.logTime = logTime;
            end

            % System parameters
            obj.sys.nArius = nArius; % number of Arius modules
            obj.sys.nChArius = 32;
            
            obj.sys.trigTxDel = 240; % [samp] trigger to t0 (tx start) delay

            obj.sys.voltage = voltage;
            
            probe = probeParams(probeName);
            obj.sys.adapType = probe.adapType;                       % 0-old(00001111); 1-new(01010101);
            obj.sys.txChannelMap = probe.txChannelMap;
            obj.sys.rxChannelMap = probe.rxChannelMap;
            obj.sys.pitch = probe.pitch;
            obj.sys.nElem = probe.nElem;
            obj.sys.xElem = (-(obj.sys.nElem-1)/2 : ...
                            (obj.sys.nElem-1)/2) * obj.sys.pitch;	% [m] (1 x nElem) x-position of probe elements

            obj.sys.nChCont = obj.sys.nChArius * (nArius*obj.sys.adapType + 1*~obj.sys.adapType);
            obj.sys.nChTotal = obj.sys.nChArius * 4 * (nArius*~obj.sys.adapType + 1*obj.sys.adapType);

            if ~obj.sys.adapType
                % old adapter type (00001111)
                obj.sys.selElem = (1:128).' + (0:(nArius-1))*128;
                obj.sys.actChan = true(128,nArius);
            else
                % new adapter type (01010101)
%                 obj.sys.selElem = reshape((1:nChan).' + (0:3)*nChan*nArius,[],1) + (0:(nArius-1))*nChan;
%                 nChanTot = nChan*4*nArius;
                obj.sys.selElem = repmat((1:128).',[1 nArius]);
                obj.sys.actChan = mod(ceil((1:128)' / nChan) - 1, nArius) == (0:(nArius-1));
            end
            obj.sys.actChan = obj.sys.actChan & (obj.sys.selElem <= obj.sys.nElem);

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
                    sequenceType = 'lin';
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
                'fsDivider', sequenceOperation.fsDivider);
            
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
                'zGrid', reconstructOperation.zGrid);
            
            obj.rec.enable = true;
            
        end
        
        function [rf,img] = run(obj)
            % Runs uploaded operations in the us4R system.
            %
            % Currently, only supports :class:`SimpleTxRxSequence` and :class:`Reconstruction`
            % implementations.
            %
            % :returns: RF frame and reconstructed image (if :class:`Reconstruction` operation was uploaded)
            
            obj.openSequence;
            rf = obj.execSequence;
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

    end

    methods(Access = private)
        % TODO:
        % Priority=Hi; usProbes.mat->function (DONE)
        % Priority=Hi; exclude calcTxParams
        % Priority=Hi; Rx aperture motion for LIN
        % Priority=Hi; Rx aperture for STA/PWI
        %               setSeqParams, calcTxParams,
        %               programHW(nSubTx),
        %               execSequence(reorganize).

        % Priority=Hi; Check the param sizes

        % Priority=Lo; scanConversion after envelope detection, scanConversion coordinates
        % Priority=Lo; Fix rounding in the aperture calculations (calcTxParams)


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
                                'fsDivider'         'fsDivider'};

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
            obj.seq.rxTime      = 160e-6; % [s] rx time (max 4000us)
            obj.seq.rxDel       = 0e-6;
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
            
            %% TGC
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
            
            %% Tx/Rx aperture positions
            if isempty(obj.seq.txApCent)
                obj.seq.txApCent	= interp1(1:obj.sys.nElem, obj.sys.xElem, obj.seq.txCentElem);
            end
            
            if isempty(obj.seq.rxApCent)
                obj.seq.rxApCent	= interp1(1:obj.sys.nElem, obj.sys.xElem, obj.seq.rxCentElem);
            else
                obj.seq.rxCentElem	= interp1(obj.sys.xElem, 1:obj.sys.nElem, obj.seq.txApCent);
            end
            
            obj.seq.rxApOrig = round(obj.seq.rxCentElem - (obj.seq.rxApSize-1)/2);

            %% Number of: Tx, SubTx, Firings, Triggers
            obj.seq.nTx	= length(obj.seq.txAng);    % could be also length(obj.seq.txApCent)
            obj.seq.nSubTx = min(4, ceil(obj.seq.rxApSize / obj.sys.nChCont));
            obj.seq.nFire = obj.seq.nTx * obj.seq.nSubTx;
            
            if isstring(obj.seq.nRep) && obj.seq.nRep == "max"
                obj.seq.nRep = min(floor([ ...
                                2^14 / obj.seq.nFire, ...
                                2^32 / obj.seq.nFire / (obj.sys.nChArius * obj.seq.nSamp * 2)]));
                disp(['nRepetitions set to ' num2str(obj.seq.nRep) '.']);
            end
            obj.seq.nTrig = obj.seq.nFire * obj.seq.nRep;

            %% Aperture masks & delays
            obj.calcTxRxApMask;
            obj.calcTxDelays;
            
            %% Piece of code moved from programHW
            nArius	= obj.sys.nArius;
            nChan	= obj.sys.nChArius;
            nSubTx	= obj.seq.nSubTx;
            nTx     = obj.seq.nTx;
            nFire	= obj.seq.nFire;
            
            txSubApDel = cell(nArius,nTx);
            txSubApMask = strings(nArius,nTx);
            rxSubApMask = strings(nArius,nFire);
            iSubTx = permute(1:nSubTx,[1 3 2]);
            for iArius=0:(nArius-1)
                txSubApDel(iArius+1,:) = mat2cell(obj.seq.txDel(obj.sys.selElem(:,iArius+1), :) .* obj.sys.actChan(:,iArius+1), 128, ones(1,nTx));
                txSubApMask(iArius+1,:) = obj.maskFormat(obj.seq.txApMask(obj.sys.selElem(:,iArius+1), :) & obj.sys.actChan(:,iArius+1));
                
                rxSubApSelect = ceil(cumsum(obj.seq.rxApMask(obj.sys.selElem(:,iArius+1), :) & obj.sys.actChan(:,iArius+1)) / nChan) == iSubTx;
                rxSubApSelect = rxSubApSelect & obj.sys.actChan(:,iArius+1);
                rxSubApMask(iArius+1,:) = obj.maskFormat(reshape(permute(obj.seq.rxApMask(obj.sys.selElem(:,iArius+1), :) & rxSubApSelect,[1 3 2]),[],nFire));
            end
            
            actChanGroupMask = obj.sys.selElem(8:8:end,:) <= obj.sys.nElem;
            actChanGroupMask = actChanGroupMask & obj.sys.actChan(8:8:end,:);
            actChanGroupMask = obj.maskFormat(actChanGroupMask);
            
            obj.seq.actChanGroupMask = actChanGroupMask;
            obj.seq.txSubApMask = txSubApMask;
            obj.seq.txSubApDel = txSubApDel;
            obj.seq.rxSubApMask = rxSubApMask;

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
                                'zGrid',            'zGrid'};

            for iPar=1:size(recParamMapping,1)
                obj.rec.(recParamMapping{iPar,2}) = [];
            end

            nPar = length(varargin)/2;
            for iPar=1:nPar
                idPar = strcmpi(varargin{iPar*2-1},recParamMapping(:,1));
                obj.rec.(recParamMapping{idPar,2}) = reshape(varargin{iPar*2},1,[]);
            end

            %% Resulting parameters
            obj.rec.zSize	= length(obj.rec.zGrid);
            obj.rec.xSize	= length(obj.rec.xGrid);

            %% Fixed parameters
            obj.rec.gpuEnable	= license('test', 'Distrib_Computing_Toolbox') && ~isempty(ver('distcomp'));
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
            % obj.seq.txApMask      - [logical] (nArius*128 x nTx) is element active in tx?
            % obj.seq.rxApMask      - [logical] (nArius*128 x nTx) is element active in rx?
            
            txApMask = abs(obj.sys.xElem' - obj.seq.txApCent) <= (obj.seq.txApSize-1)/2*obj.sys.pitch;
            rxApMask = abs(obj.sys.xElem' - obj.seq.rxApCent) <= (obj.seq.rxApSize-1)/2*obj.sys.pitch;
            
            % Make the mask fit the number of channels
            if obj.sys.nElem >= obj.sys.nChTotal
                txApMask = txApMask(1:obj.sys.nChTotal,:);
                rxApMask = rxApMask(1:obj.sys.nChTotal,:);
            else
                txApMask = [txApMask; false(obj.sys.nChTotal-obj.sys.nElem, obj.seq.nTx)];
                rxApMask = [rxApMask; false(obj.sys.nChTotal-obj.sys.nElem, obj.seq.nTx)];
            end
            
            % Save the mask to the obj
            obj.seq.txApMask = txApMask;
            obj.seq.rxApMask = rxApMask;
        end

        function calcTxDelays(obj)
            % calcTxDelays appends the following fields to the in/out obj:
            % obj.seq.txDel         - [s] (nArius*128 x nTx) tx delays for each element
            % obj.seq.txDelCent     - [s] (1 x nTx) tx delays for tx aperture centers
            
            %% CALCULATE DELAYS
            if isinf(obj.seq.txFoc)
                % Delays due to the tilting the plane wavefront
                txDel       = (obj.sys.xElem.'  .* sin(obj.seq.txAng) ) / obj.seq.c;	% [s] (nElem x nTx) delays for tx elements
                txDelCent	= (obj.seq.txApCent .* sin(obj.seq.txAng) ) / obj.seq.c;	% [s] (1 x nTx) delays for tx aperture center

            else
                % Focal point positions
                xFoc        = obj.seq.txFoc .* sin(obj.seq.txAng) + obj.seq.txApCent;	% [m] (1 x nTx) x-position of the focal point
                zFoc        = obj.seq.txFoc .* cos(obj.seq.txAng);                      % [m] (1 x nTx) z-position of the focal point

                % Delays due to the element - focal point distances
                txDel       = sqrt((xFoc - obj.sys.xElem.' ).^2 + zFoc.^2) / obj.seq.c;	% [s] (nElem x nTx) delays for tx elements
                txDelCent	= sqrt((xFoc - obj.seq.txApCent).^2 + zFoc.^2) / obj.seq.c;	% [s] (1 x nTx) delays for tx aperture center

                % Inverse the delays for the 'focusing' option (zFoc>0)
                % For 'defocusing' the delays remain unchanged
                focDefoc	= 1 - 2*max(0,sign(zFoc));
                txDel       = txDel     .* focDefoc;
                txDelCent	= txDelCent .* focDefoc;
            end

            %% Postprocess the delays
            % Make the delays fit the number of channels
            txDel       = [txDel;	 zeros(obj.sys.nArius*128-obj.sys.nElem, obj.seq.nTx)];
            
            % Make delays = nan outside the tx aperture
            txDel(~obj.seq.txApMask)	= nan;

            % Make delays >= 0 in the tx aperture
            txDelShift	= - nanmin(txDel);              % [s] (1 x nTx)
            txDel       = txDel     + txDelShift;       % [s] (nElem x nTx)
            txDelCent	= txDelCent + txDelShift;       % [s] (1 x nTx)

            % Equalize the txCentDel
            txDel       = txDel - txDelCent + max(txDelCent);
            txDelCent	= max(txDelCent);

            % Remove nans
            txDel(~obj.seq.txApMask)	= 0;

            %% Save the delays to the obj
            obj.seq.txDel       = txDel;
            obj.seq.txDelCent	= txDelCent;

        end
        
        function validateSequence(obj)
            
            %% Validate number of firings
            if obj.seq.nFire > 1024
                error("ARRUS:IllegalArgument", ...
                        ['Number of firings (' num2str(obj.seq.nFire) ') cannot exceed 1024.' ]);
            end
            
            %% Validate number of triggers
            if obj.seq.nTrig > 16384
                error("ARRUS:IllegalArgument", ...
                        ['Number of triggers (' num2str(obj.seq.nTrig) ') cannot exceed 16384.']);
            end
            
            %% Validate number of samples
            if obj.seq.nSamp > 2^13/obj.seq.fsDivider
                error("ARRUS:IllegalArgument", ...
                        ['Number of samples ' num2str(obj.seq.nSamp) ' cannot exceed ' num2str(2^13/obj.seq.fsDivider) '.'])
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
            
            %% Program mappings, gains, and voltage
            for iArius=0:(obj.sys.nArius-1)
                % Set Rx channel mapping
                for ch=1:32
                    Us4MEX(iArius, "SetRxChannelMapping", obj.sys.rxChannelMap(iArius+1,ch), ch);
                end

                % Set Tx channel mapping
                for ch=1:128
                    Us4MEX(iArius, "SetTxChannelMapping", obj.sys.txChannelMap(iArius+1,ch), ch);
                end

                % init RX
                Us4MEX(iArius, "SetPGAGain","30dB");
                Us4MEX(iArius, "SetLPFCutoff","15MHz");
                Us4MEX(iArius, "SetActiveTermination","EN", "200");
                Us4MEX(iArius, "SetLNAGain","24dB");
                Us4MEX(iArius, "SetDTGC","DIS", "0dB");                 % EN/DIS? (attenuation actually, 0:6:42)
                Us4MEX(iArius, "TGCEnable");

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
            end
            
            %% Program Tx/Rx sequence
            for iArius=0:(obj.sys.nArius-1)
                for iFire=0:(obj.seq.nFire-1)
                    %% active channel groups
                    Us4MEX(iArius, "SetActiveChannelGroup", obj.seq.actChanGroupMask(iArius+1), iFire);
                    
                    %% Tx
                    iTx     = 1 + floor(iFire/obj.seq.nSubTx);
                    Us4MEX(iArius, "SetTxAperture", obj.seq.txSubApMask(iArius+1,iTx), iFire);
                    Us4MEX(iArius, "SetTxDelays", obj.seq.txSubApDel{iArius+1,iTx}, iFire);
                    Us4MEX(iArius, "SetTxFrequency", obj.seq.txFreq, iFire);
                    Us4MEX(iArius, "SetTxHalfPeriods", obj.seq.txNPer*2, iFire);
                    Us4MEX(iArius, "SetTxInvert", 0, iFire);
                    
                    %% Rx
                    Us4MEX(iArius, "SetRxAperture", obj.seq.rxSubApMask(iArius+1,iFire+1), iFire);
                    Us4MEX(iArius, "SetRxTime", obj.seq.rxTime, iFire);
                    Us4MEX(iArius, "SetRxDelay", obj.seq.rxDel, iFire);
                    Us4MEX(iArius, "TGCSetSamples", obj.seq.tgcCurve, iFire);
                end
                Us4MEX(iArius, "SetNumberOfFirings", obj.seq.nFire);
                Us4MEX(iArius, "EnableTransmit");
                Us4MEX(iArius, "EnableReceive");
            end
            
            %% Program triggering
            Us4MEX(0, "SetNTriggers", obj.seq.nTrig);
            for iTrig=0:(obj.seq.nTrig-1)
                Us4MEX(0, "SetTrigger", obj.seq.txPri*1e6, 0, 0, iTrig);
            end
            Us4MEX(0, "SetTrigger", obj.seq.txPri*1e6, 0, 1, obj.seq.nTrig-1);
            for iArius=1:(obj.sys.nArius-1)
                Us4MEX(iArius, "SetTrigger", obj.seq.txPri*1e6, 0, 0, 0);
            end
            
            %% Program recording
            for iArius=0:(obj.sys.nArius-1)
                Us4MEX(iArius, "ClearScheduledReceive");
                for iTrig=0:(obj.seq.nTrig-1)
                    Us4MEX(iArius, "ScheduleReceive", iTrig*obj.seq.nSamp, obj.seq.nSamp, obj.seq.startSample + obj.sys.trigTxDel, obj.seq.fsDivider-1);
                end
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

        function rf = execSequence(obj)

            nArius	= obj.sys.nArius;
            nChan	= obj.sys.nChArius;
            nSamp	= obj.seq.nSamp;
            nSubTx	= obj.seq.nSubTx;
            nTx     = obj.seq.nTx;
            nRep	= obj.seq.nRep;
            nTrig	= nTx*nSubTx*nRep;

            %% Capture data
            for iArius=0:(nArius-1)
                Us4MEX(iArius, "EnableReceive");
            end
            Us4MEX(0, "TriggerSync");
            pause(obj.seq.pauseMultip * obj.seq.txPri * nTrig);

            %% Transfer to PC
            rf = Us4MEX(0, ...
                        "TransferAllRXBuffersToHost",  ...
                        zeros(nArius, 1), ...
                        repmat(nSamp * nTrig, [nArius 1]), ...
                        int8(obj.logTime) ...
            );

            %% Reorganize
            rf	= reshape(rf, [nChan, nSamp, nSubTx, nTx, nRep, nArius]);

            rxApOrig = obj.seq.rxApOrig;
            if ~obj.sys.adapType
                rf	= permute(rf,[2 1 3 6 4 5]);
                
                % old adapter type (00001111)
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
                rf = reshape(rf,nSamp,nChan*nSubTx,nArius,nTx,nRep);
                rfAux = permute(rf,[1 2 4 5 3]);
                rf = zeros(nSamp,obj.seq.rxApSize,nTx,nRep);
                for iTx=1:nTx
                    rxApEnd = rxApOrig(iTx) + obj.seq.rxApSize - 1;
                    nZerosL = max(0, min(  0,rxApEnd) -          rxApOrig(iTx)  + 1);
                    nZerosR = max(0,         rxApEnd  - max(193, rxApOrig(iTx)) + 1);
                    nChan0  = max(0, min(128,rxApEnd) - max(  1, rxApOrig(iTx)) + 1);
                    nChan1  = max(0, min(192,rxApEnd) - max(129, rxApOrig(iTx)) + 1);
                    
                    rf(:,:,iTx,:) = [zeros(nSamp,nZerosL,1,nRep), ...
                        rfAux(:,1:nChan0,iTx,:,1), ...
                        rfAux(:,1:nChan1,iTx,:,2), ...
                        zeros(nSamp,nZerosR,1,nRep)];
                end
                
            else
                % new adapter type (01010101)
                rf	= permute(rf,[2 1 6 3 4 5]);
                rf	= reshape(rf,nSamp,nChan*nArius,nSubTx,nTx,nRep);
                
                for iTx=1:nTx
                    rf(:,:,:,iTx,:)	= circshift(rf(:,:,:,iTx,:),-mod(rxApOrig(iTx)-1,nChan*nArius),2);
                end
                rf	= reshape(rf,nSamp,nChan*nArius*nSubTx,nTx,nRep);
                rf	= rf(:,1:obj.seq.rxApSize,:,:);
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

            % Digital Down Conversion
            rfRaw = downConversion(rfRaw,obj.seq,obj.rec);

            % warning: both filtration and decimation introduce phase delay!
            % rfRaw = preProc(rfRaw,obj.seq,obj.rec);

            %% Reconstruction
            if strcmp(obj.seq.type,'lin')
                rfBfr = reconstructRfLin(rfRaw,obj.sys,obj.seq,obj.rec);
            else
                rfBfr = reconstructRfImg(rfRaw,obj.sys,obj.seq,obj.rec);
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
                envImg = scanConversion(envImg,obj.seq,obj.rec);
            end
            
            % Compression
            img = 20*log10(envImg);
            
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
        
    end
end
