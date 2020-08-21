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
            % The method programHW() is for hardware programming.
                        
            % The method propertiesPreprocessing() below 
            %   enumerates following properties:
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
%             nRxChannels = obj.usSystem.nChArius*3; % max number of rx channels 
            samplingFrequency = 65e6;
            
            % time need for switch from transmit to receive or vice-versa
            trSwitchTime = 240./samplingFrequency; 
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
                
%                 % Set Rx channel mapping 
                for iFire = 0:obj.nFire-1
                    Us4MEX(iArius, "SetRxChannelMapping", ...
                        obj.usSystem.rxChannelMap(iArius+1,1:32), ...
                        iFire ...
                        );
%                     disp(obj.usSystem.rxChannelMap(iArius+1,1:32))
                end
                
                % Set Tx channel mapping
                for iChannel = 1:nTxChannels
                    Us4MEX(iArius, "SetTxChannelMapping", ...
                       obj.usSystem.txChannelMap(iArius+1, iChannel), ...
                       iChannel ...
                       );
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
            for iArius = 0:nArius-1
                Us4MEX(iArius, "SetNumberOfFirings", obj.nFire);
                Us4MEX(iArius, "ClearScheduledReceive"); 
            end

            iFire = 0;
            nSamp = NaN(1,obj.nFire);
            startSamp = NaN(1,obj.nFire);
            for iTxRx = 1:nTxRx 
                rxTime = obj.sequence.TxRxList(iTxRx).Rx.delay  ...
                    + obj.sequence.TxRxList(iTxRx).Rx.time ...
                    + trSwitchTime ...
                    ;                
                fs = samplingFrequency./obj.sequence.TxRxList(iTxRx).Rx.fsDivider;
                moduleTxApertures = obj.module2TxMaps{iTxRx};
                moduleTxDelays = obj.module2TxDelaysMaps{iTxRx};
                moduleRxApertures = obj.module2RxMaps{iTxRx};                
                for iSubTxRx = 1:obj.nSubTxRx(iTxRx)

                    nSamp(iFire+1) = floor(obj.sequence.TxRxList(iTxRx).Rx.time.*fs);
                    startSamp(iFire+1) = floor(obj.sequence.TxRxList(iTxRx).Rx.delay.*fs);

%                     disp(obj.sequence.TxRxList(iTxRx).Rx.delay)
%                     disp(startSamp)
                    
                    for iArius = 0:nArius-1
%                         disp(actChanGroupMask(iArius+1))
                        Us4MEX(iArius, "SetActiveChannelGroup", actChanGroupMask(iArius+1), iFire);

                        % Tx
                        Us4MEX(iArius, "SetTxAperture", obj.maskFormat(moduleTxApertures(iArius+1,:).'), iFire);
                        Us4MEX(iArius, "SetTxDelays", moduleTxDelays(iArius+1,:), iFire);
                        Us4MEX(iArius, "SetTxFrequency", obj.sequence.TxRxList(iTxRx).Tx.pulse.frequency, iFire);
                        Us4MEX(iArius, "SetTxHalfPeriods", obj.sequence.TxRxList(iTxRx).Tx.pulse.nPeriods*2, iFire);
                        Us4MEX(iArius, "SetTxInvert", 0, iFire);

                        % Rx
%                         disp(obj.maskFormat(moduleRxApertures(iArius+1, :, iSubTxRx).'))
%                         disp(obj.sequence.TxRxList(iTxRx).Rx.delay)

%                         Us4MEX(iArius, "SetRxChannelMapping", ...
%                             obj.usSystem.rxChannelMap(iArius+1,1:32), ...
%                             iFire ...
%                             );

                        Us4MEX(iArius, "SetRxAperture", obj.maskFormat(moduleRxApertures(iArius+1, :, iSubTxRx).'), iFire);
                        Us4MEX(iArius, "SetRxTime", rxTime, iFire); 
                        Us4MEX(iArius, "SetRxDelay", obj.sequence.TxRxList(iTxRx).Rx.delay, iFire);
                        % do zrobienia tgc
%                         Us4MEX(iArius, "TGCSetSamples", obj.seq.tgcCurve, iFire);
%                         Us4MEX(iArius, "ClearScheduledReceive"); 
                        Us4MEX(iArius, "ScheduleReceive", ...
                            iFire, ...
                            iFire*nSamp(iFire+1), ...
                            nSamp(iFire+1), ... 
                            startSamp(iFire+1) + obj.usSystem.trigTxDel, ...
                            obj.sequence.TxRxList(iTxRx).Rx.fsDivider-1,...
                            iFire ...
                            );

                    end
                    iFire = iFire+1;
                end
                
            end
            % note: after loop over n TxRx events the 'iFire' represents
            % total number of firings
            obj.nSamp = nSamp;
            obj.startSamp = startSamp;
%             disp(['nSamp: ',num2str(nSamp)])
%             disp(['startSamp: ',num2str(startSamp)])

            
            for iArius = 0:nArius-1
                Us4MEX(iArius, "EnableTransmit");
            end
            
           

            % Program triggering
            for iArius = 0:obj.usSystem.nArius-1
                Us4MEX(iArius, "SetNTriggers", obj.nFire);
                for iTrig = 0:obj.nFire-2
%                     Us4MEX(iArius, "SetTrigger", obj.sequence.pri*1e6,  0, iTrig);
                    Us4MEX(iArius, "SetTrigger", obj.sequence.TxRxList(iTrig+1).pri*1e6,  0, iTrig);
                end
%                 Us4MEX(iArius, "SetTrigger", obj.sequence.pri*1e6, 1, obj.nFire-1);
                Us4MEX(iArius, "SetTrigger", obj.sequence.TxRxList(nFire).pri*1e6, 1, obj.nFire-1);
                Us4MEX(iArius, "EnableSequencer");
            end

        end
        
        
        function propertiesPreprocessing(obj)
            nTxRx = length(obj.sequence.TxRxList);
            nSubTxRx = NaN(1,nTxRx);
            for i = 1:nTxRx 

                
                % interpretation of empty properties in TxRx objects

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
            
            % if Rx.aperture is all false, then nSumTxRx is 0, but should
            % be 1 for Us4MEX, because must be nFire <=1
            if isequal(nSubTxRx,0)
                obj.nSubTxRx = 1;
            else
                obj.nSubTxRx = nSubTxRx;
            end
            
            obj.nFire = sum(obj.nSubTxRx);

            
        end % of propertiesPreprocessing()
                
        
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
            
        end % of maskFormat()
        
 
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
%             nModules = 2; 
            nModules = obj.usSystem.nArius;
            
            % number of channels in module
%             nModuleChannels = 128; 
            nModuleChannels = obj.usSystem.nChTotal./obj.usSystem.nArius; 
            
            % number of available rx channels in single module            
%             nRxChannerls = 32; 
            nRxChannels = obj.usSystem.nChArius;
            
            % number of rx channel groups
            nRxChanGroups = 3; 
            
            
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
            
            %{
            % clear empty channel groups (i.e. size(moduleRxApertures,3)
            % will be equal to nFire
            emptyGroups = [];
            for iRxChanGroup = 1:nRxChanGroups
               if isempty(find(moduleRxApertures(:,:,iRxChanGroup), 1))
                   emptyGroups = [emptyGroups,iRxChanGroup];
               end               
            end
            moduleRxApertures(:,:,emptyGroups) = [];
            %}
            
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
        
        
            rf = obj.reshapeMexRf(rf);
            
            % Stop acquisition
            Us4MEX(0, "TriggerStop");

        end % of run()
    
        
        function rfRshpd = reshapeMexRf(obj, rf)
            % The method reshapes rf array 
            % from UsRMEX(iModule, "TransferAllRXBuffersToHost", ...
            % (size nChannels x nAllSamp, where nChannels is a number 
            % of available rx channels in a single module)
            % to final rf array (size nSamp x nElements)

            [nChannels, nAllSampn] = size(rf);
            nElement = obj.usSystem.nElem;
            nTxRx = length(obj.sequence.TxRxList);
            nModule = obj.usSystem.nArius;

            rfRshpd = zeros(max(obj.nSamp), nElement, nTxRx);

            maps = obj.module2RxMaps;
            sample0 = 0;
            for iModule = 1:nModule
                iFire = 0;
                for iTxRx = 1:nTxRx
                   map = maps{iTxRx};
                   
                   for iSubTxRx = 1:obj.nSubTxRx(nTxRx)
                      iFire = iFire+1;
                      samples = (1:obj.nSamp(iFire))+sample0;
                      sample0 = sample0+obj.nSamp(iFire);
                      
                      % indexes of probe elements corresponding to
                      % subaperture of Rx.aperture
                      activeElements = map(iModule, :, iSubTxRx); 
                      activeElements(activeElements==0)=[];
                      activeChannels = mod(activeElements-1,nChannels)+1;

                      if ~isempty(activeElements)
                          rfRshpd(:,activeElements, iTxRx) = rf(activeChannels, samples).';
                      end
                       
                   end
                end
                
            end
            

        end
        
    end
   
end