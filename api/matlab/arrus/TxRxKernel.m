classdef TxRxKernel
    % Class for ultrasound system programming when tx-rx sequence 
    % is defined via object of TxRxSequence class.
    %
    %
    % properties:
    %   sequence - TxRxsequence class object (or empty array)
    %   usSystem - sys structure, for now internal structure in Us4R.
    %
    % methods:
    %   TxRxKernel() - class constructor. 
    %       To pass arguments to the constructor name-value convetion 
    %           is used,
    %
    %   programHW(obj) - program the US system,
    %   calcNFire(obj) - calculate the number of firings used by sequence.
    
    properties
        sequence = TxRxSequence()
        usSystem = []
    end
    
    methods
        function obj = TxRxKernel(varargin)
            if nargin ~= 0
                p = inputParser;
                                
                % adding parameters to parser
                addParameter(p, 'usSystem', [])
                addParameter(p, 'sequence', TxRxSequence())
                parse(p, varargin{:})
                
                obj.usSystem = p.Results.usSystem;
                obj.sequence= p.Results.sequence;
            end
            
            
        end
        
        
        function programHW(obj)
            
%             nArius = obj.usSystem.nArius; % number of arius modules
%             nRxChannels = obj.usSystem.nChArius; % max number of rx channels 
            nTxChannels = 128; % max number of tx channels

            nTxRx = length(obj.sequence.TxRxList);
            
            actChanGroupMask = obj.usSystem.selElem(8:8:end,:) <= obj.usSystem.nElem;
            actChanGroupMask = actChanGroupMask & obj.usSystem.actChan(8:8:end,:);
            actChanGroupMask = obj.maskFormat(actChanGroupMask);

          
            
            
            
            % Program mappings, gains, and voltage
            for iArius = 0:nArius-1
                
                % Set Rx channel mapping
                for iChannel = 1:nRxChannels
                    Us4MEX(iArius, "SetRxChannelMapping", ...
                           obj.usSystem.rxChannelMap(iArius+1, iChannel), ...
                           iChannel);
                end

                % Set Tx channel mapping
                for iChannel = 1:nTxChannels
                    Us4MEX(iArius, "SetTxChannelMapping", ...
                           obj.usSystem.txChannelMap(iArius+1, iChannel), ...
                           iChannel);
                end

                % init RX
                Us4MEX(iArius, "SetPGAGain","30dB");
                Us4MEX(iArius, "SetLPFCutoff","15MHz");
                Us4MEX(iArius, "SetActiveTermination","EN", "200");
                Us4MEX(iArius, "SetLNAGain","24dB");
                Us4MEX(iArius, "SetDTGC","DIS", "0dB");
                Us4MEX(iArius, "TGCEnable");

                try
                    Us4MEX(0,"EnableHV");
                    
                catch
                    warning('1st "EnableHV" failed');
                    Us4MEX(0,"EnableHV");
                    
                end
                
                try
                    Us4MEX(0, "SetHVVoltage", obj.usSystem.voltage);
                    
                catch
                    warning('1st "SetHVVoltage" failed');
                    Us4MEX(0, "SetHVVoltage", obj.usSystem.voltage);
                    
                end
            end
            
            
            
            % Program Tx/Rx sequence
            for i = 1:nTxRx
                thisTxRx = obj.sequence.TxRxList(i);
                txAp = thisTxRx.Tx.aperture;
                rxAp = thisTxRx.Rx.aperture;
                txDel = thisTxRx.Tx.delay;
                rxDel = thisTxRx.Rx.delay;
                rxTime = thisTxRx.Rx.time;
                
                [moduleTxApertures, moduleTxDelays, moduleRxApertures] = apertures2modules(txAp, txDel, rxAp);
                nFire = size(moduleRxApertures,3); 
                
                
                for iFire = 0:nFire-1
                    for iArius = 0:nArius-1
                        % active channel groups
                        Us4MEX(iArius, "SetActiveChannelGroup", actChanGroupMask(iArius+1), iFire);


                        % Tx
    %                     iTx     = 1 + floor(iFire/nSubTx);
                        Us4MEX(iArius, "SetTxAperture", moduleTxApertures(iArius+1,:), iFire);
                        Us4MEX(iArius, "SetTxDelays", moduleTxDelays(iArius+1,:), iFire);


                        Us4MEX(iArius, "SetTxFrequency", thisTxRx.Tx.pulse.frequency, iFire);
                        Us4MEX(iArius, "SetTxHalfPeriods", thisTxRx.Tx.pulse.nPeriods, iFire);
                        Us4MEX(iArius, "SetTxInvert", 0, iFire);

                        % Rx
                        Us4MEX(iArius, "SetRxAperture", moduleRxApertures(iArius, :, iFire), iFire);
                        Us4MEX(iArius, "SetRxTime", rxTime, iFire);
                        Us4MEX(iArius, "SetRxDelay", rxDel, iFire);

                    end

                end
                
                for iArius = 0:nArius-1
                    Us4MEX(iArius, "SetNumberOfFirings", nFire);
                    Us4MEX(iArius, "EnableTransmit");
                    Us4MEX(iArius, "EnableReceive");
                end

                
            end
            
  
     % TUTAJ TRIGERY ZROBIC DOBRZE!!!
            % Program triggering
            Us4MEX(0, "SetNTriggers", nFire);
            for iTrig = 0:nFire-1
                Us4MEX(0, "SetTrigger", obj.sequence.pri*1e6, 0, 0, iTrig);
            end
            Us4MEX(0, "SetTrigger", obj.sequence.pri*1e6, 0, 1, obj.seq.nTrig-1);
            for iArius = 1:obj.usSystem.nArius-1
                Us4MEX(iArius, "SetTrigger", obj.sequence.pri*1e6, 0, 0, 0);
            end
            
            % Program recording
            for iArius=0:(obj.usSystem.nArius-1)
                Us4MEX(iArius, "ClearScheduledReceive");
                
                for iTrig=0:(obj.seq.nTrig-1)
                    Us4MEX(iArius, "ScheduleReceive", ...
                        iTrig*obj.seq.nSamp, ...
                        obj.seq.nSamp, ...
                        obj.seq.startSample + obj.sys.trigTxDel, ...
                        obj.seq.fsDivider-1 ...
                    );
                end
                
            end
            
            %}
            
        end
        
        
        function nFire = calcNFire(obj)
            % The method calculate the number of firings neccessary to
            % realize the TxRxSequence
            
%             maxRxChannels = obj.sys.nChArius;
            maxRxChannels = 32;
            nFire = 0;
            for iTxRx = 1:length(obj.sequence.TxRxList)
                thisRxAperture = obj.sequence.TxRxList(1,iTxRx).Rx.aperture;
                apertureLenght = length(thisRxAperture);
                iTxRxFirings = ceil(apertureLenght./maxRxChannels);
                nFire = nFire + iTxRxFirings;

            end
        end % of calcNFire()
        

        
        
        function maskString = maskFormat(maskLogical)
            
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
        
        
        
        
        function [moduleTxApertures, moduleTxDelays, moduleRxApertures] = apertures2modules(txAp, txDel, rxAp)
            
        % The method maps logical transmit aperture, transmit delays 
        %   and receive aperture into mask array
        % 
        % It returns 3 arrays:
        %   moduleTxApertures, moduleTxDelays are of size [nModules, nModuleChannels]
        %   moduleRxApertures, is of size [nModules, nModuleChannels, nFire]
        %   where nFire is the number of firings necessary to acquire
        %   rxAperture.
        
        
            nModules = 2; % number of modules 
            nModuleChannels = 128; % number of channels in module
            nRxChannels = 32; % number of available rx channels in single module
            nRxChanGroups = 3; % number of rx channel groups
            
            % some validation - not sure if it is necessary (should be
            % checked later)
            if length(txAp) > nModules*nModuleChannels
               error('Transmit aperture length is bigger than number of available channels.') 
            end
            
            
            % moduleChannel-to-apertureElement mapping
            module2elementArray = zeros(nModules,nRxChannels,nRxChanGroups);
            module2elementArray(1, 1:32, 1) = 0+(1:32);
            module2elementArray(1, 1:32, 2) = 64+(1:32);
            module2elementArray(1, 1:32, 3) = 128+(1:32);
            module2elementArray(2, 1:32, 1) = 32+(1:32);
            module2elementArray(2, 1:32, 2) = 96+(1:32);
            module2elementArray(2, 1:32, 3) = 160+(1:32);
            
            % TX PART
            
            % allocation of tx output arrays
            moduleTxApertures = false(nModules, nModuleChannels);
            moduleTxDelays = zeros(nModules, nModuleChannels);
            
            % mapping tx arrays
            for iChannel = 1:length(txAp)
                for iModule = 1:nModules
                    if txAp(iChannel)==1 && ismember(iChannel, module2elementArray(iModule,:,:)) 
                        moduleTxApertures(iModule, iChannel) = true;
                        moduleTxDelays(iModule, iChannel) = txDel(iChannel);
                    end
                end
            end
            
            % RX PART            

            % allocation of rx output arrays            
            moduleRxApertures = false(nModules,nModuleChannels,nRxChanGroups);
            
            % mapping rx array
            for iChannel = 1:length(rxAp)
                for iModule = 1:nModules
                    for iRxChanGroup = 1:nRxChanGroups
%                         iRxChannel = mod(iChannel,nRxChannels+1)+(floor(iChannel/(nRxChannels+1)));
                        if rxAp(iChannel)==1 && ismember(iChannel, module2elementArray(iModule,:,iRxChanGroup)) 
                            moduleRxApertures(...
                                iModule, ...
                                iChannel, ...
                                iRxChanGroup ...
                                ) = true;
                        end
                    end
                end
            end

            % clear empty channel groups (i.e. size(moduleRxApertures,3)
            % will be equal to nFire
            emptyGroups = [];
            for iRxChanGroup = 1:nRxChanGroups
               if isempty(find(moduleRxApertures(:,:,iRxChanGroup), 1))
                   emptyGroups = [emptyGroups,iRxChanGroup];
               end               
            end
            moduleRxApertures(:,:,emptyGroups) = [];
            
        end % of apertures2modules()     
        

        
    end
   
end