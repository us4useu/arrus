classdef TxRxKernel < handle
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
    
    properties (Access = private)
        
        nFire = 0 % number of all firings
        nSamp = 0 % vector with sample numbers for all firings
        startSamp = 0 % vector with first sample numbers for all firings
        nSubTxRx = 0 % vector with number of firings in each TxRx event. sum(nSubTxRx) == nFire
        
        % Arrays describing relation between module channel 
        % and probe element in each TxRx, used in programHW
        module2RxMaps = {} 
        module2TxMaps = {}
        module2TxDelaysMaps = {}

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
                        
            % The method below enumerated following properties:
            %   nFire - number of all firings
            %   nSubTxRx -  vector with number of firings in each TxRx event. 
            %        sum(nSubTxRx) == nFire
            %   module2RxMaps, module2TxMaps, module2TxDelaysMaps
            %       - Arrays describing relation between module channel 
            %         and probe element in each TxRx
            obj.propertiesPreprocessing()
            
            nArius = obj.usSystem.nArius; % number of arius modules
%             nRxChannels = obj.usSystem.nChArius; % max number of rx channels
            % new firmware
            nRxChannels = obj.usSystem.nChArius*3; % max number of rx channels 
            samplingFrequency = 65e6;
            nTxChannels = 128; % max number of tx channels
            nTxRx = length(obj.sequence.TxRxList);
            
            actChanGroupMask = obj.usSystem.selElem(8:8:end,:) <= obj.usSystem.nElem;
            actChanGroupMask = actChanGroupMask & obj.usSystem.actChan(8:8:end,:);
            actChanGroupMask = obj.maskFormat(actChanGroupMask);

            % Unloading Us4MEX should clear the device state.
            munlock('Us4MEX');
            clear Us4MEX;            
            
            % Program mappings, gains, and voltage
            for iArius = 0:nArius-1
                
%                 % Set Rx channel mapping (old firmware)
                for iFire = 0:obj.nFire-1

%                     for iChannel = 1:nRxChannels
%                         Us4MEX(iArius, "SetRxChannelMapping", ...
%                                obj.usSystem.rxChannelMap(iArius+1, iChannel), ...
%                                iChannel, ...
%                                iFire...
%                                );
%                     end
                    % new firmware
%                     disp(obj.usSystem.rxChannelMap(iArius+1,1:32))
                    Us4MEX(iArius, "SetRxChannelMapping", ...
                        obj.usSystem.rxChannelMap(iArius+1,1:32), iFire ...
                        );
                end

                % Set Rx channel mapping (new firmware)Us4MEX(ar-1, "SetTxChannelMapping", esaote_mapping(ch,ar)+1, ch);
%                                            obj.usSystem.rxChannelMap(:, :)
%                 for iChannel = 1:nRxChannels
%                     Us4MEX(iArius, "SetRxChannelMapping", ...
%                            obj.usSystem.rxChannelMap(iArius+1, iChannel), ...
%                            iChannel ...
%                            );
%                 end

                
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
                Us4MEX(iArius, "TGCDisable");

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

            % new firmware
            for iArius = 0:nArius-1
                Us4MEX(iArius, "SetNumberOfFirings", obj.nFire);
                Us4MEX(iArius, "ClearScheduledReceive"); 
            end

            iFire = 0;
            nSamp = NaN(1,obj.nFire);
            startSamp = NaN(1,obj.nFire);
            for iTxRx = 1:nTxRx 
                fs = samplingFrequency./obj.sequence.TxRxList(iTxRx).Rx.fsDivider;
                moduleTxApertures = obj.module2TxMaps{iTxRx};
                moduleTxDelays = obj.module2TxDelaysMaps{iTxRx};
                moduleRxApertures = obj.module2RxMaps{iTxRx};                
                for iSubTxRx = 0:obj.nSubTxRx(iTxRx)-1

                    nSamp(iFire+1) = floor(obj.sequence.TxRxList(iTxRx).Rx.time.*fs);
                    startSamp(iFire+1) = floor(obj.sequence.TxRxList(iTxRx).Rx.delay./fs);

                    for iArius = 0:nArius-1
%                         Us4MEX(iArius, "ClearScheduledReceive"); % new firmware
                        % active channel groups
%                         disp(actChanGroupMask(iArius+1))
                        Us4MEX(iArius, "SetActiveChannelGroup", actChanGroupMask(iArius+1), iFire);
                        
%                         disp(['module ',num2str(iArius),', txaperture'])
%                         disp(['subfire no. ',num2str(iTxRxFire)])
%                         disp(obj.maskFormat(moduleTxApertures(iArius+1,:).'))
                        % Tx
                        Us4MEX(iArius, "SetTxAperture", obj.maskFormat(moduleTxApertures(iArius+1,:).'), iFire);
                        Us4MEX(iArius, "SetTxDelays", moduleTxDelays(iArius+1,:), iFire);
                        Us4MEX(iArius, "SetTxFrequency", obj.sequence.TxRxList(iTxRx).Tx.pulse.frequency, iFire);
                        Us4MEX(iArius, "SetTxHalfPeriods", obj.sequence.TxRxList(iTxRx).Tx.pulse.nPeriods, iFire);
                        Us4MEX(iArius, "SetTxInvert", 0, iFire);

%                         disp(['module ',num2str(iArius),', rxaperture'])
%                         disp(['subfire no. ',num2str(iTxRxFire)])
%                         disp(obj.maskFormat(moduleRxApertures(iArius+1, :, iTxRxFire+1).'))
                        % Rx
                        Us4MEX(iArius, "SetRxAperture", obj.maskFormat(moduleRxApertures(iArius+1, :, iSubTxRx+1).'), iFire);
                        Us4MEX(iArius, "SetRxTime", obj.sequence.TxRxList(iTxRx).Rx.time, iFire);
                        Us4MEX(iArius, "SetRxDelay", obj.sequence.TxRxList(iTxRx).Rx.delay, iFire);
                        % do zrobienia tgc
%                         Us4MEX(iArius, "TGCSetSamples", obj.seq.tgcCurve, iFire);

                    end
                    iFire = iFire+1;
                end
                
            end
            % note: after loop over n TxRx events the 'iFire' represents
            % total number of firings
            obj.nSamp = nSamp;
            obj.startSamp = startSamp;

            
            for iArius = 0:nArius-1
%                 Us4MEX(iArius, "SetNumberOfFirings", obj.nFire);
                Us4MEX(iArius, "EnableTransmit");
%                 Us4MEX(iArius, "EnableReceive");
            end
            
  
     
            % Program triggering
            Us4MEX(0, "SetNTriggers", obj.nFire);
            for iTrig = 0:obj.nFire-1% -2?
%                 Us4MEX(0, "SetTrigger", obj.sequence.pri*1e6, 0, 0, iTrig);
                Us4MEX(0, "SetTrigger", obj.sequence.pri*1e6,  0, iTrig);
            end
            Us4MEX(0, "SetTrigger", obj.sequence.pri*1e6, 1, obj.nFire-1);
            for iArius = 1:obj.usSystem.nArius-1
                Us4MEX(iArius, "SetTrigger", obj.sequence.pri*1e6, 0, 0);
            end
            

            % Program recording
            for iArius = 0:obj.usSystem.nArius-1
                Us4MEX(iArius, "ClearScheduledReceive"); % po co to, skro bylo wczesniej (przed definiowaniem txrx?
                
                iTrig = 0;
                for iTxRx = 1:length(obj.sequence.TxRxList)
                    for iSubTxRx = 1:obj.nSubTxRx(iTxRx)
                        iTrig = iTrig+1;
%                         % old firmware
%                         Us4MEX(iArius, "ScheduleReceive", ...
%                             (iTrig-1)*obj.nSamp(iTrig), ...
%                             obj.nSamp(iTrig), ...
%                             obj.startSamp(iTrig) + obj.usSystem.trigTxDel, ...
%                             obj.sequence.TxRxList(iTxRx).Rx.fsDivider - 1 ...  
%                         );
                        % new firmware
                        Us4MEX(...
                            iArius, "ScheduleReceive", ...
                            (iTrig-1), ...
                            (iTrig-1)*obj.nSamp(iTrig), ...
                            obj.nSamp(iTrig), ... 
                            obj.startSamp(iTrig) + obj.usSystem.trigTxDel, ...
                            obj.sequence.TxRxList(iTxRx).Rx.fsDivider - 1, ...  
                            (iTrig-1) ...
                            );
                        
                    end
                    
                end
                

            end            

        end
        
        
        function propertiesPreprocessing(obj)
            nTxRx = length(obj.sequence.TxRxList);
            nSubTxRx = NaN(1,nTxRx);
            for i = 1:nTxRx 
                thisTxRx = obj.sequence.TxRxList(i);
                
                %% interpretation of empty properties in TxRx objects

                if isempty(obj.sequence.TxRxList(i).Tx.aperture)
                    obj.sequence.TxRxList(i).Tx.aperture = ...
                        false(1,obj.usSystem.nElem);
                end
                
                if isempty(obj.sequence.TxRxList(i).Rx.aperture)
                    obj.sequence.TxRxList(i).Rx.aperture = ...
                        false(1,obj.usSystem.nElem);
                end  
                
                if isempty(obj.sequence.TxRxList(i).Tx.delay)
                    obj.sequence.TxRxList(i).Tx.delay = ...
                        zeros(1,obj.usSystem.nElem);
                end
                
                if isempty(obj.sequence.TxRxList(i).Rx.delay)
                    obj.sequence.TxRxList(i).Rx.delay = 0;
                end
                
                
                if isempty(obj.sequence.TxRxList(i).Rx.time)
                    obj.sequence.TxRxList(i).Rx.time = 0;
                end
                
                if isempty(obj.sequence.TxRxList(i).Rx.fsDivider)
                    obj.sequence.TxRxList(i).Rx.fsDivider = 1;
                end
                
                if isempty(obj.sequence.TxRxList(i).Tx.pulse)
                    pulse = Pulse;
                    pulse.frequency = 0;
                    pulse.nPeriods = 0;
                    obj.sequence.TxRxList(i).Tx.pulse = pulse;
                end
                
                [moduleTxApertures, moduleTxDelays, moduleRxApertures] = ...
                    obj.apertures2modules( ...
                        obj.sequence.TxRxList(i).Tx.aperture, ...
                        obj.sequence.TxRxList(i).Tx.delay, ...
                        obj.sequence.TxRxList(i).Rx.aperture ...
                        );
                    

                obj.module2RxMaps{i} = moduleRxApertures;
                obj.module2TxMaps{i} = moduleTxApertures;
                obj.module2TxDelaysMaps{i} = moduleTxDelays;
                nTxRxFire = size(moduleRxApertures,3); % number of fires for this single TxRx 
                nSubTxRx(i) = nTxRxFire; % number of subFirings will be used later in run() method
               
            end
            obj.nSubTxRx = nSubTxRx;
            obj.nFire = sum(nSubTxRx);
            
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
        

        
        
        function maskString = maskFormat(obj, maskLogical)
            maskLogical = logical(maskLogical);
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
        
 
        function [moduleTxApertures, moduleTxDelays, moduleRxApertures] = apertures2modules(obj, txAp, txDel, rxAp)
            
        % The method maps logical transmit aperture, transmit delays 
        %   and receive aperture into mask corresponding to module
        %   channels.
        % 
        % It returns 3 arrays:
        %   moduleTxApertures, moduleTxDelays are of size [nModules, nModuleChannels]
        %   moduleRxApertures, is of size [nModules, nModuleChannels, nFire]
        %   where nFire is the number of firings necessary to acquire
        %   rxAperture.
        
        % number of modules 
            nModules = 2; 
%             nModules = obj.usSystem.nArius;
            
            % number of channels in module
            nModuleChannels = 128; 
%             nModuleChannels = obj.usSystem.nChTotal./usSystem.nArius; 
            
            % number of available rx channels in single module            
            nRxChannels = 32; 
%             nRxChannels = obj.usSystem.nChArius;
            
            % number of rx channel groups
            nRxChanGroups = 3; 
            
            % some validation - not sure if it is necessary (should be
            % checked later)
            if length(txAp) > nModules*nModuleChannels
               error('Transmit aperture length is bigger than number of available channels.') 
            end
            
            % Creating array which maps module channels into probe elements:
            %   array indexes corresponds to iModule, iRxChannel and
            %   iRxGhanGroup while values corresponds to element numbers
            
            % allocation
            module2elementArray = zeros(nModules,nRxChannels,nRxChanGroups);
            elements0 = zeros(nModules, nRxChanGroups);
            
            % first-1 elements of groups operates by module 1
            elements0(1,:) = [0,64,128];
           
            % first-1 elements of groups operates by module 2
            elements0(2,:) = [32,96,160];
            
            for iModule = 1:nModules
                for iGroup = 1:nRxChanGroups
                    iElement0 = elements0(iModule, iGroup);
                    module2elementArray(iModule, 1:nRxChannels, iGroup) = ...
                        (1:nRxChannels)+iElement0;
                end
            end

            
            
            
            

            % TX PART
            
            % allocation of tx output arrays i.e. aperture masks and delays
            % for modules (used by Us4MEX())
            moduleTxApertures = zeros(nModules, nModuleChannels);
            moduleTxDelays = zeros(nModules, nModuleChannels);

            % mapping tx module apertures
            for iElement = 1:length(txAp)
                for iModule = 1:nModules
                    if txAp(iElement)==1 
                        [iRxChannel, iRxChanGroup] = ...
                            find(squeeze(module2elementArray(iModule,:,:)) == iElement);
                        if ~isempty(iRxChannel)
                            iModuleChannel = iRxChannel+(iRxChanGroup-1)*nRxChannels;
                            moduleTxApertures(iModule, iModuleChannel) = iElement;
                            moduleTxDelays(iModule, iModuleChannel) = txDel(iElement);
                        end
                    end
                end
            end
%             moduleTxApertures
            



            % RX PART            

            % allocation of rx output arrays            
            moduleRxApertures = zeros(nModules,nModuleChannels,nRxChanGroups);
            % mapping rx array
            for iElement = 1:length(rxAp)
                for iModule = 1:nModules
                    for iRxChanGroup = 1:nRxChanGroups
%                         iRxChannel = mod(iChannel,nRxChannels+1)+(floor(iChannel/(nRxChannels+1)));
                        if rxAp(iElement)==1 %&& ismember(iElement, module2elementArray(iModule,:,iRxChanGroup)) 
                            iRxChannel = ...
                                find(squeeze(module2elementArray(iModule,:,iRxChanGroup)) == iElement);
                            if ~isempty(iRxChannel)
                                iModuleChannel = iRxChannel+(iRxChanGroup-1)*nRxChannels;
                                moduleRxApertures(...
                                    iModule, ...
                                    iModuleChannel, ...
                                    iRxChanGroup ...
                                    ) = iElement;
                            end
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
        
        
        function rf = run(obj)
            pauseMultip = 2;
          
            % Start acquisitions (1st sequence exec., no transfer to host)
            Us4MEX(0, "TriggerStart");
            pause(pauseMultip * obj.sequence.pri * obj.nFire);

            %% Capture data
%             for iArius=0:(obj.usSystem.nArius-1)
%                 Us4MEX(iArius, "EnableReceive");
%             end
            Us4MEX(0, "TriggerSync");
            pause(pauseMultip * obj.sequence.pri * obj.nFire);
            
            %% Transfer to PC

            nAllSamp = sum(obj.nSamp);
            rf = Us4MEX(0, ...
                        "TransferAllRXBuffersToHost",  ...
                        zeros(obj.usSystem.nArius, 1), ...
                        repmat(nAllSamp, [obj.usSystem.nArius, 1]), ...
                        int8(1) ...
            );
        
        
%             rf = obj.reshapeMexRf(rf);
            
            % Stop acquisition
            Us4MEX(0, "TriggerStop");

        end % of run
        
        function rfRshpd = reshapeMexRf(obj, rf)
            rf = rf.';
            nElement = obj.usSystem.nElem;
            [nAllSampn, Channels ] = size(rf);
            nSamp = obj.nSamp;
            nFire = obj.nFire;
            nTxRx = length(obj.sequence.TxRxList);
            nModule = obj.usSystem.nArius;

            
            rfRshpd = zeros(max(nSamp), nElement, nTxRx);
            
            maps = obj.module2RxMaps;
            sample0 = 0;
            for iModule = 1:nModule
                iFire = 0;
                for iTxRx = 1:nTxRx
                   map = maps{iTxRx};
                   [~,~,nSubTxRx] = size(map);
                   for iSubTxRx = 1:nSubTxRx
                      iFire = iFire+1;
                      samples = (1:nSamp(iFire))+sample0;
                      sample0 = sample0+nSamp(iFire);
                      
                      elements = map(iModule, :, iSubTxRx);
                      elements(elements==0)=[];
                      rfRshpd(:,elements, iTxRx) = rf(samples,:);
                       
                   end
                end
                
            end
            

        end
        
    end
   
end