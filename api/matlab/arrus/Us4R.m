classdef Us4R < handle
    % A handle to the Us4R system. 
    %
    % This class provides functions to configure the system and perform
    % data acquisition using the Us4R.
    %
    % Please note: only one instance of this class can be in use at a time!
    
    properties(Access = private)
        sys
        seq
    end
    
    methods

        function obj = Us4R(nArius,probeName)
            % Us4R handle constructor.
            %
            % :param nArius: number of arius modules available in the \
            %  us4R system
            % :param probeName: probe name to use
            % :returns: Us4R instance

            % System parameters
            obj.sys.nArius = nArius; % number of Arius modules
            obj.sys.nChArius = 32;

            probe = probeParams(probeName);
            obj.sys.adapType = probe.adapType;                       % 0-old(00001111); 1-new(01010101);
            obj.sys.pitch = probe.pitch;
            obj.sys.nElem = probe.nElem;
            obj.sys.xElem = (-(obj.sys.nElem-1)/2 : ...
                            (obj.sys.nElem-1)/2) * obj.sys.pitch;	% [m] (1 x nElem) x-position of probe elements

            for iArius=0:(nArius-1)
                % Set Rx channel mapping
                for ch=1:32
                    AriusMEX(iArius, "SetRxChannelMapping", probe.rxChannelMap(iArius+1,ch), ch);
                end

                % Set Tx channel mapping
                for ch=1:128
                    AriusMEX(iArius, "SetTxChannelMapping", probe.txChannelMap(iArius+1,ch), ch);
                end

                % init RX
                AriusMEX(iArius, "SetPGAGain","30dB");
                AriusMEX(iArius, "SetLPFCutoff","15MHz");
                AriusMEX(iArius, "SetActiveTermination","EN", "200");
                AriusMEX(iArius, "SetLNAGain","24dB");
                AriusMEX(iArius, "SetDTGC","DIS", "0dB");                 % EN/DIS? (attenuation actually, 0:6:42)
                AriusMEX(iArius, "TGCSetSamples", uint16([hex2dec('9001'), hex2dec('4000')+(3000:-75:0), hex2dec('4000')+3000]));
                AriusMEX(iArius, "TGCEnable");

                try
                    AriusMEX(0,"EnableHV");
                catch
                    warning('1st "EnableHV" failed');
                    AriusMEX(0,"EnableHV");
                end

                AriusMEX(0,"SetHVVoltage", 10);
            end

        end

        function obj = upload(obj, operation)
            % Uploads operation to the us4R system.
            %
            % Currently, only supports :class:`SimpleTxRxSequence`
            % implementations.
            %
            % :param operation: operation to perform on the us4R system
            % :returns: updated Us4R object
            
            switch(class(operation))
                case 'PWISequence'
                    sequenceType = "pwi";
                case 'STASequence'
                    sequenceType = "sta";
                case "LINSequence"
                    sequenceType = 'lin';
                otherwise
                    error("ARRUS:IllegalArgument", ...
                        ['Unrecognized operation type ', class(operation)])
            end
            
            obj.setSeqParams(...
                'sequenceType', sequenceType, ...
                'txCenterElement', operation.txCenterElement, ...
                'txApertureCenter', operation.txApertureCenter, ...
                'txApertureSize', operation.txApertureSize, ...
                'txFocus', operation.txFocus, ...
                'txAngle', operation.txAngle, ...
                'speedOfSound', operation.speedOfSound, ...
                'txFrequency', operation.txFrequency, ...
                'txNPeriods', operation.txNPeriods, ...
                'rxNSamples', operation.rxNSamples, ...
                'txPri', operation.txPri);
            
        end
        
        function rf = run(obj)
            % Runs uploaded operation in the us4R system.
            %
            % Currently, only supports :class:`SimpleTxRxSequence`
            % implementations.
            %
            % :param operation: operation to perform on the us4R system
            % :returns: RF frame
            
            obj.openSequence;
            rf = obj.execSequence;
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
                                'txFocus',          'txFoc'; ...
                                'txAngle',          'txAng'; ...
                                'speedOfSound',     'c'; ...
                                'txFrequency',      'txFreq'; ...
                                'txNPeriods',       'txNPer'; ...
                                'rxNSamples',       'nSamp'; ...
                                'txPri',            'txPri'};

            if mod(length(varargin),2) == 1
                % TODO(piotrkarwat) Throw exception
            end

            for iPar=1:size(seqParamMapping,1)
                eval(['obj.seq.' seqParamMapping{iPar,2} ' = [];']);
            end

            nPar = length(varargin)/2;
            for iPar=1:nPar
                idPar = find(strcmpi(varargin{iPar*2-1},seqParamMapping(:,1)));

                if isempty(idPar)
                    % TODO(piotrkarwat) Throw exception
                end

                if ~isnumeric(varargin{iPar*2})
                    % TODO(piotrkarwat) Throw exception
                end

                eval(['obj.seq.' seqParamMapping{idPar,2} ' = reshape(varargin{iPar*2},1,[]);']);
            end

            %% Resulting parameters
            if isempty(obj.seq.txApCent) && ~isempty(obj.seq.txCentElem)
                obj.seq.txApCent        = obj.sys.xElem(floor(obj.seq.txCentElem)).*(1-mod(obj.seq.txCentElem,1)) + ...
                                          obj.sys.xElem( ceil(obj.seq.txCentElem)).*(  mod(obj.seq.txCentElem,1));
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

            if strcmp(obj.seq.type,'lin')
                obj.seq.nSubTx          = 1;
            else
                if ~obj.sys.adapType
                    % old adapter type (00001111)
                    obj.seq.nSubTx      = min(4, ceil(obj.sys.nElem / obj.sys.nChArius));
                else
                    % new adapter type (01010101)
                    obj.seq.nSubTx      = min(4, ceil(obj.sys.nElem / (obj.sys.nChArius * obj.sys.nArius)));
                end
            end

            %% Fixed parameters
            obj.seq.rxSampFreq	= 65e6;                                 % [Hz] sampling frequency
            obj.seq.rxTime      = 160e-6;                                % [s] rx time (max 4000us)
            obj.seq.rxDel       = 5e-6;
            obj.seq.pauseMultip	= 1.5;

            %% Program hardware
            obj	= obj.programHW;

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

            txDel       = [txDel;	 zeros(obj.sys.nArius*128-obj.sys.nElem, obj.seq.nTx)];
            txApMask	= [txApMask; false(obj.sys.nArius*128-obj.sys.nElem, obj.seq.nTx)];

            %% Save the apertures and delays to the obj
            obj.seq.txApMask	= txApMask;
            obj.seq.txDel       = txDel;
            obj.seq.txDelCent	= txDelCent;

        end

        function obj = programHW(obj)

            nArius	= obj.sys.nArius;
            nSamp	= obj.seq.nSamp;
            nChan	= obj.sys.nChArius;
            nSubTx	= obj.seq.nSubTx;
            nTx     = obj.seq.nTx;
            nEvent	= nSubTx*nTx;

            %% Program TX
            for iArius=0:(nArius-1)
                for iTx=1:nTx
                    for iSubTx=1:nSubTx
                        iEvent	= iSubTx-1 + (iTx-1)*nSubTx;

                        if ~obj.sys.adapType
                            % old adapter type (00001111)
                            selectElem	= (1:128) + iArius*128;
                        else
                            % new adapter type (01010101)
                            selectElem	= reshape((1:nChan)' + (0:3)*nChan*nArius,1,[]) + iArius*nChan;
                        end
                        txSubApDel	= obj.seq.txDel(selectElem,iTx);
                        txSubApSize	= sum(obj.seq.txApMask(selectElem, iTx));
                        txSubApOrig	= find(obj.seq.txApMask(selectElem, iTx),1,'first');
                        if isempty(txSubApOrig)
                            txSubApOrig = 1;
                        end

                        AriusMEX(iArius, "SetTxAperture", txSubApOrig, txSubApSize, iEvent);
                        AriusMEX(iArius, "SetTxDelays", txSubApDel, iEvent);

                        AriusMEX(iArius, "SetTxFrequency", obj.seq.txFreq, iEvent);
                        AriusMEX(iArius, "SetTxHalfPeriods", obj.seq.txNPer*2, iEvent);
                        AriusMEX(iArius, "SetTxInvert", 0, iEvent);
                    end
                end
                AriusMEX(iArius, "SetNumberOfFirings", nEvent);
                AriusMEX(iArius, "EnableTransmit");
            end

            %% Program RX
            if strcmp(obj.seq.type,'lin')
                obj.seq.rxApOrig	= nan(1,nTx);
            end

            for iArius=0:(nArius-1)
                AriusMEX(iArius, "ClearScheduledReceive");
                for iTx=1:nTx

                    if strcmp(obj.seq.type,'lin') && iArius == 0
                        rxCentElem	= interp1(obj.sys.xElem,1:obj.sys.nElem,obj.seq.txApCent(iTx));
                        if ~obj.sys.adapType
                            % old adapter type (00001111)
                            obj.seq.rxApOrig(iTx)	= round(rxCentElem - (nChan-1)/2);
                        else
                            % new adapter type (01010101)
                            obj.seq.rxApOrig(iTx)	= round(rxCentElem - (nChan*nArius-1)/2);
                        end
                    end

                    for iSubTx=1:nSubTx
                        iEvent	= iSubTx-1 + (iTx-1)*nSubTx;

                        AriusMEX(iArius, "ScheduleReceive", iEvent*nSamp, nSamp);

                        AriusMEX(iArius, "SetRxTime", obj.seq.rxTime, iEvent);
                        AriusMEX(iArius, "SetRxDelay", obj.seq.rxDel, iEvent);

                        if strcmp(obj.seq.type,'lin')
                            if ~obj.sys.adapType
                                % old adapter type (00001111)
                                rxSubApOrig	= obj.seq.rxApOrig(iTx) - 4*nChan*iArius;
                            else
                                % new adapter type (01010101)
                                rxSubApOrig	= obj.seq.rxApOrig(iTx) - nChan*iArius;
                                rxSubApOrig	= 1 + min(nChan,mod(rxSubApOrig-1,nChan*nArius)) ...
                                                + nChan*floor((rxSubApOrig-1)/(nChan*nArius));
                            end

                            rxSubApSize	= max(0, min([nChan, nChan + rxSubApOrig - 1, 4*nChan - rxSubApOrig + 1]));
                            rxSubApOrig	= max(1, min(4*nChan,rxSubApOrig));
                        else
                            rxSubApOrig	= 1 + (iSubTx-1)*nChan;
                            rxSubApSize	= nChan;
                        end
                        AriusMEX(iArius, "SetRxAperture", rxSubApOrig, rxSubApSize, iEvent);
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



        function [] = openSequence(obj)
            nSubTx	= obj.seq.nSubTx;
            nTx     = obj.seq.nTx;
            nEvent	= nTx*nSubTx;

            %% Start acquisitions (1st sequence exec., no transfer to host)
            AriusMEX(0, "TriggerStart");
            pause(obj.seq.pauseMultip * obj.seq.txPri*1e-6 * nEvent);
        end

        function [] = closeSequence(obj)
            %% Stop acquisition
            AriusMEX(0, "TriggerStop");

        end

        function rf = execSequence(obj)

            nArius	= obj.sys.nArius;
            nChan	= obj.sys.nChArius;
            nSamp	= obj.seq.nSamp;
            nSubTx	= obj.seq.nSubTx;
            nTx     = obj.seq.nTx;
            nEvent	= nTx*nSubTx;

            %% Capture data
            for iArius=0:(nArius-1)
                AriusMEX(iArius, "EnableReceive");
            end
            AriusMEX(0, "TriggerSync");
            pause(obj.seq.pauseMultip * obj.seq.txPri*1e-6 * nEvent);

            %% Transfer to PC
            rf	= zeros(nChan,nSamp*nEvent,nArius);
            for iArius=0:(nArius-1)
                rf(:,:,iArius+1)	= AriusMEX(iArius, "TransferRXBufferToHost", 0, nSamp * nEvent);
            end

            %% Reorganize
            rf	= reshape(rf, [nChan, nSamp, nSubTx, nTx, nArius]);

            if ~obj.sys.adapType
                % old adapter type (00001111)
                rf	= permute(rf,[2 1 3 5 4]);
                rf	= reshape(rf,nSamp,nChan*nSubTx*nArius,nTx);
            else
                % new adapter type (01010101)
                rf	= permute(rf,[2 1 5 3 4]);
                rf	= reshape(rf,nSamp,nChan*nArius*nSubTx,nTx);
            end

            if strcmp(obj.seq.type,'lin')
                rxApOrig	= obj.seq.rxApOrig;
                if ~obj.sys.adapType
                    % old adapter type (00001111)
                    for iTx=1:nTx
                        rf(:,:,iTx)	= circshift(rf(:,:,iTx),-min(32,max(0,rxApOrig(iTx)-1-nChan*(4-1))),2);
                    end
                    rf	= rf(:,1:nChan,:);
                    for iTx=1:nTx
                        if ~(rxApOrig(iTx) > 1+nChan*(4-1) && rxApOrig(iTx) < 1+nChan*4)
                            rf(:,:,iTx)	= circshift(rf(:,:,iTx),-mod(rxApOrig(iTx)-1,nChan),2);
                        end
                    end
                else
                    % new adapter type (01010101)
                    for iTx=1:nTx
                        rf(:,:,iTx)	= circshift(rf(:,:,iTx),-mod(rxApOrig(iTx)-1,nChan*nArius),2);
                    end
                end
            else
                rf	= rf(:,1:min(obj.sys.nElem,nChan*nSubTx*nArius),:);
            end

        end
    end
end
