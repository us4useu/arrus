% class for system control
classdef us4R
    
    
    properties %(Access = private)
        
        sys
        seq
        rec
        
    end
    
    
    methods
        
        % To do: 
        % Priority=Hi; Rx aperture motion for LIN
        % Priority=Lo; Check the param sizes
        % Priority=Lo; Fix rounding in the aperture calculations (calcTxParams)
        
        function obj = us4R(nArius,probeName)
            
%             path(path, '..\..\x64\Release');
            addpath('C:\arius\matlab');
            
            % load probe parameters
            load('usProbes.mat','probes');
            probesAux	= struct2cell(probes(:));
            iProbe      = find(strcmpi(probesAux(1,:),probeName),1);
            if isempty(iProbe)
                % Throw exception
            end
            
            % System parameters
            obj.sys.nArius      = nArius;                               % number of Arius modules
            obj.sys.nChArius	= 32;
            obj.sys.pitch       = probes(iProbe).pitch;
            obj.sys.nElem       = probes(iProbe).nElements;
            obj.sys.xElem       = (-(obj.sys.nElem-1)/2 : ...
                                    (obj.sys.nElem-1)/2) * obj.sys.pitch;	% [m] (1 x nElem) x-position of probe elements
            
            for iArius=0:(nArius-1)
                % Set Rx channel mapping
                for ch=1:32
                    AriusMEX(iArius, "SetRxChannelMapping", probes(iProbe).rxChannelMap(iArius+1,ch), ch);
                end
                
                % Set Tx channel mapping
                for ch=1:128
                    AriusMEX(iArius, "SetTxChannelMapping", probes(iProbe).txChannelMap(iArius+1,ch), ch);
                end
                
                % init RX
                AriusMEX(iArius, "SetPGAGain","30dB");
                AriusMEX(iArius, "SetLPFCutoff","15MHz");
                AriusMEX(iArius, "SetActiveTermination","EN", "200");
                AriusMEX(iArius, "SetLNAGain","24dB");
                AriusMEX(iArius, "SetDTGC","DIS", "0dB");                 % EN/DIS? (attenuation actually, 0:6:42)
                AriusMEX(iArius, "TGCSetSamples", uint16([hex2dec('9001'), hex2dec('4000')+(3000:-75:0), hex2dec('4000')+3000]));
                AriusMEX(iArius, "TGCEnable");
                
                AriusMEX(0,"EnableHV");
                AriusMEX(0,"SetHVVoltage", 10);
            end
            
        end
        
        
        
        function obj = setSeqParams(obj,varargin)
            
            %% Set sequence parameters
            % Sequence parameters names mapping
            %                    public name         private name
            seqParamMapping = { 'sequenceType',     'type'
                                'txCenterElement',  'txCentElem'; ...
                                'txApertureCenter', 'txApCent'; ...
                                'txApertureSize',   'txApSize'; ...
                                'rxApertureSize',   'rxApSize'; ...
                                'txFocus',          'txFoc'; ...
                                'txAngle',          'txAng'; ...
                                'speedOfSound',     'c'; ...
                                'txFrequency',      'txFreq'; ...
                                'txNPeriods',       'txNPer'; ...
                                'rxNSamples',       'nSamp'; ...
                                'txPri',            'txPri'};
            
            if mod(length(varargin),2) == 1
                % Throw exception
            end
            
            for iPar=1:size(seqParamMapping,1)
                eval(['obj.seq.' seqParamMapping{iPar,2} ' = [];']);
            end
            
            nPar = length(varargin)/2;
            for iPar=1:nPar
                idPar = find(strcmpi(varargin{iPar*2-1},seqParamMapping(:,1)));
                
                if isempty(idPar)
                    % Throw exception
                end
                
                if ~isnumeric(varargin{iPar*2})
                    % Throw exception
                end
                
                eval(['obj.seq.' seqParamMapping{idPar,2} ' = reshape(varargin{iPar*2},1,[]);']);
            end
            
            %% Resulting parameters
            if isempty(obj.seq.txApCent) && ~isempty(obj.seq.txCentElem)
                obj.seq.txApCent        = obj.seq.xElem(floor(obj.seq.txCentElem))*(1-mod(obj.seq.txCentElem,1)) + ...
                                          obj.seq.xElem( ceil(obj.seq.txCentElem))*(  mod(obj.seq.txCentElem,1));
            end
            
            switch obj.seq.type
                case 'sta'
                    obj.seq.nTx         = length(obj.seq.txApCent);
                case 'pwi'
                    obj.seq.nTx         = length(obj.seq.txAng);
                case 'lin'
                    obj.seq.nTx         = length(obj.seq.txApCent);
            end
            
            obj = obj.calcTxParams;
            
            %% Fixed parameters
            obj.seq.nSubTx      = 4;
            obj.seq.rxSampFreq	= 65e6;                                 % [Hz] sampling frequency
            obj.seq.rxTime      = 160e-6;                                % [s] rx time (max 4000us)
            obj.seq.rxDel       = 5e-6;
            %% Program hardware
            obj.programHW;
            
        end
        
        
        
        function obj = setRecParams(obj,varargin)
            
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
            
            if mod(length(varargin),2) == 1
                % Throw exception
            end
            
            for iPar=1:size(recParamMapping,1)
%                 eval(['obj.seq.' recParamMapping{iPar,2} ' = [];']);
                eval(['obj.rec.' recParamMapping{iPar,2} ' = [];']);
            end
            
            nPar = length(varargin)/2;
            for iPar=1:nPar
                idPar = find(strcmpi(varargin{iPar*2-1},recParamMapping(:,1)));
                
                if isempty(idPar)
                    % Throw exception
                end
                
                if ~isnumeric(varargin{iPar*2})
                    % Throw exception
                end
                
                eval(['obj.rec.' recParamMapping{idPar,2} ' = reshape(varargin{iPar*2},1,[]);']);
            end
            
            %% Resulting parameters
            obj.rec.zSize	= length(obj.rec.zGrid);
            obj.rec.xSize	= length(obj.rec.xGrid);
            
            %% Fixed parameters
            obj.rec.gpuEnable	= license('test', 'Distrib_Computing_Toolbox') && ~isempty(ver('distcomp'));
            
        end
        
        
        
        function val = get(obj,paramName)
            
            if isfield(obj.sys,paramName)
                val = eval(['obj.sys.' paramName]);
            else
                if isfield(obj.seq,paramName)
                    val = eval(['obj.seq.' paramName]);
                else
                    if isfield(obj.rec,paramName)
                        val = eval(['obj.rec.' paramName]);
                    else
                        error('Invalid parameter name');
                    end
                end
            end
            
        end
        
        
        
        function obj = calcTxParams(obj)
            % calcTxParams appends the following fields to the in/out obj:
            % obj.seq.txApMask      - [logical] (nArius*128 x nTx) is element active in tx?
            % obj.seq.txDel         - [s] (nArius*128 x nTx) tx delays for each element
            % obj.seq.txDelCent     - [s] (1 x nTx) tx delays for tx aperture centers
            
            %% CALCULATE APERTURE MASKS
            %Rounding!!!
            txApMask	= abs(obj.sys.xElem' - obj.seq.txApCent) <= (obj.seq.txApSize-1)/2*obj.sys.pitch;
            
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
            
            % Make delays = nan outside the tx aperture
            txDel(~txApMask)	= nan;
            
            % Make delays >= 0 in the tx aperture
            txDelShift	= - nanmin(txDel);              % [s] (1 x nTx)
            txDel       = txDel     + txDelShift;       % [s] (nElem x nTx)
            txDelCent	= txDelCent + txDelShift;       % [s] (1 x nTx)
            
            % Equalize the txCentDel
            txDel       = txDel - txDelCent + max(txDelCent);
            txDelCent	= max(txDelCent);
            
            %% Make the apertures/delays fit the number of channels
            txDel(~txApMask)	= 0;
            
            txApMask	= [txApMask; false(obj.sys.nArius*128-obj.sys.nElem, obj.seq.nTx)];
            txDel       = [txDel;	 zeros(obj.sys.nArius*128-obj.sys.nElem, obj.seq.nTx)];
            
            %% Save the apertures and delays to the obj
            obj.seq.txApMask	= txApMask;
            obj.seq.txDel       = txDel;
            obj.seq.txDelCent	= txDelCent;
            
        end
        
        
        
        function [] = programHW(obj)
            
            nArius	= obj.sys.nArius;
            nSamp	= obj.seq.nSamp;
            nSubTx	= obj.seq.nSubTx;
            nTx     = obj.seq.nTx;
            nEvent	= nSubTx*nTx;
            
            %% Program TX
            for iArius=0:(nArius-1)
                for iTx=1:nTx
                    for iSubTx=1:nSubTx
                        iEvent	= iSubTx-1 + (iTx-1)*nSubTx;
                        
                        txSubApSize	= sum(obj.seq.txApMask((1:128)+iArius*128, iTx));
                        txSubApOrig	= find(obj.seq.txApMask((1:128)+iArius*128, iTx),1,'first');
                        if isempty(txSubApOrig)
                            txSubApOrig = 1;
                        end
                        
                        AriusMEX(iArius, "SetTxAperture", txSubApOrig, txSubApSize, iEvent);
                        AriusMEX(iArius, "SetTxDelays", obj.seq.txDel((1:128)+128*iArius,iTx), iEvent);
                        
                        AriusMEX(iArius, "SetTxFrequency", obj.seq.txFreq, iEvent);
%                         AriusMEX(iArius, "SetTxPeriods", obj.seq.txNPer, iEvent);
                        AriusMEX(iArius, "SetTxHalfPeriods", obj.seq.txNPer*2, iEvent);
                        AriusMEX(iArius, "SetTxInvert", 0, iEvent);
                    end
                end
                AriusMEX(iArius, "SetNumberOfFirings", nEvent);
                AriusMEX(iArius, "EnableTransmit");
            end
            
            %% Program RX
            for iArius=0:(nArius-1)
                AriusMEX(iArius, "ClearScheduledReceive");
                for iTx=1:nTx
                    for iSubTx=1:nSubTx
                        iEvent	= iSubTx-1 + (iTx-1)*nSubTx;
                        
                        AriusMEX(iArius, "ScheduleReceive", iEvent*nSamp, nSamp);
                        
                        AriusMEX(iArius, "SetRxTime", obj.seq.rxTime, iEvent);
                        AriusMEX(iArius, "SetRxDelay", obj.seq.rxDel, iEvent);
                        
                        rxSubApSize	= obj.sys.nChArius;
                        rxSubApOrig	= 1 + (iSubTx-1)*obj.sys.nChArius;
                        AriusMEX(iArius, "SetRxAperture", rxSubApOrig, rxSubApSize, iEvent);
%                         AriusMEX(iArius, "SetRxAperture", (iTx-1)*32+1, 32, iEvent);
                    end
                end
                AriusMEX(iArius, "EnableReceive");
            end
            
            %% Program triggering
%             AriusMEX(0, "SetNTriggers", nEvent-1);
            AriusMEX(0, "SetNTriggers", nEvent);
            for iEvent=0:(nEvent-1)
                AriusMEX(0, "SetTrigger", obj.seq.txPri, 0, 0, iEvent);
            end
            AriusMEX(0, "SetTrigger", obj.seq.txPri, 0, 1, nEvent-1);
            
        end
        
        
        
        function rf = execSequence(obj)
            
            nArius	= obj.sys.nArius;
            nChan	= obj.sys.nChArius;
            nSamp	= obj.seq.nSamp;
            nSubTx	= obj.seq.nSubTx;
            nTx     = obj.seq.nTx;
            nEvent	= nTx*nSubTx;
            
            %% Capture data
%             AriusMEX(0, "SWTrigger");
            
            AriusMEX(0, "TriggerStart");
            pause(2*obj.seq.txPri*1e-6*nEvent);
            if 0
                for iArius=0:(nArius-1)
                    AriusMEX(iArius, "EnableReceive");
                end
                AriusMEX(0, "TriggerSync");
                pause(2*obj.seq.txPri*1e-6*nEvent);
                
                %% Transfer to PC
                rf	= zeros(nChan,nSamp*nEvent,nArius);
                for iArius=0:(nArius-1)
                    rf(:,:,iArius+1)	= AriusMEX(iArius, "TransferRXBufferToHost", 0, nSamp * nEvent);
                end
            end
            AriusMEX(0, "TriggerStop");
            
            %% Transfer to PC
            rf	= zeros(nChan,nSamp*nEvent,nArius);
            for iArius=0:(nArius-1)
                rf(:,:,iArius+1)	= AriusMEX(iArius, "TransferRXBufferToHost", 0, nSamp * nEvent);
            end
            
            %% Reorganize
            rf	= reshape(rf, [nChan, nSamp, nSubTx, nTx, nArius]);
            rf	= permute(rf,[2 1 3 5 4]);
            rf	= reshape(rf,nSamp,nChan*nSubTx*nArius,nTx);
            rf	= rf(:,1:min(obj.sys.nElem,nChan*nSubTx*nArius),:);
            
        end
        
        
        
        function img = execReconstr(obj,rfRaw)
            
            %% Move data to GPU if possible
            if obj.rec.gpuEnable
                rfRaw = gpuArray(rfRaw);
            end
            
            %% Preprocessing
            % Raw rf data filtration
            if obj.rec.filtEnable
                rfRaw = filter(obj.rec.filtB,obj.rec.filtA,rfRaw);
            end
            
            % Digital Down Conversion
            rfRaw = downConv(rfRaw,obj.seq,obj.rec);
            
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
            
            % Scan conversion (for 'lin' mode)
            if strcmp(obj.seq.type,'lin')
                rfBfr = scanConv(rfBfr,obj.sys,obj.seq,obj.rec);
            end
            
            % Envelope detection
            envImg = abs(rfBfr);
            
            % Compression
            img = 20*log10(envImg);
            
            % Gather data from GPU
            if obj.rec.gpuEnable
                img = gather(img);
            end
            
            
        end
        
        
        
        function [] = runOnce(obj)
            
            %% TX/RX sequence
            rf	= obj.execSequence;
            
            %% Reconstruction
            img	= obj.execReconstr(rf);
            
            %% Display
            if obj.rec.filtEnable
                rf = filter(obj.rec.filtB,obj.rec.filtA,rf);
            end
            
            nTx     = obj.seq.nTx;
            
            figure;
            for iTx=1:nTx
                subplot(1,nTx,iTx);
                imagesc(rf(:,:,iTx));
                xlabel('Chan #');
                ylabel('Samp #');
                colormap(jet);
                colorbar;
                set(gca,'CLim',[-1 1]*1e2);
            end
            set(gcf,'Position',get(gcf,'Position') + [560 0 0 0]);
            
            figure;
            imagesc(obj.rec.xGrid*1e3,obj.rec.zGrid*1e3,img);
            xlabel('x [mm]');
            ylabel('z [mm]');
            daspect([1 1 1]);
%             set(gca,'CLim',[40 80]);
            colormap(gray);
            colorbar;
            
        end
        
        
        
        function [] = runLoop(obj)
            
            %% Prepare the display
            img     = zeros(obj.rec.zSize,obj.rec.xSize);
            hFig	= figure;
            hImg	= imagesc(obj.rec.xGrid*1e3,obj.rec.zGrid*1e3,img);
            xlabel('x [mm]');
            ylabel('z [mm]');
            daspect([1 1 1]);
            set(gca,'CLim',[-20 80]);
            colormap(gray);
            colorbar;
            
            %% TX/RX / Reconstruction / Display
            while(ishghandle(hFig))
                % Display
                tic;
                set(hImg, 'CData', img);
                drawnow;
                disp(['Disp. time = ' num2str(toc,'%5.3f') 's']);
                disp('-------------------');
                
                % TX/RX sequence
                tic;
                rf	= obj.execSequence;
                disp(['Acq. time  = ' num2str(toc,'%5.3f') 's']);
                
                % Reconstruction
                tic;
                img	= obj.execReconstr(rf);
                disp(['Rec. time  = ' num2str(toc,'%5.3f') 's']);
            end
            
        end
        
        
        
    end
end
