% txAp = true(1,33);
% rxAp = true(1,33);
% txDel = zeros(1,33);

% txAp = [true(1,33), false(1,64), true(1,31)];
% txAp = true(1,192);
% rxAp = true(1,192);
% txDel = zeros(size(txAp));


clc
tx = Tx();
rx = Rx();
txrx = TxRx('Tx', Tx, 'Rx', Rx, 'pri', 1e-6)
seq = TxRxSequence([txrx, txrx,txrx])


% [moduleTxApertures, moduleTxDelays, moduleRxApertures] = apertures2modules(txAp, txDel, rxAp);

% a = [1:32,65:96, 129:160]
% b = [33:64,97:128,161:192]


function [moduleTxApertures, moduleTxDelays, moduleRxApertures] = apertures2modules(txAp, txDel, rxAp)
            
        % The method maps logical transmit aperture, transmit delays 
        %   and receive aperture into mask array
        % 
        % It returns 3 arrays:
        %   moduleTxApertures, moduleTxDelays are of size [nModules, nModuleChannels]
        %   moduleRxApertures, is of size [nModules, nModuleChannels, nFire]
        %   where nFire is the number of firings necessary to acquire
        %   rxAperture.
        
        % number of modules 
            nModules = 2; 
%             nModules = usSystem.nArius;
            
            % number of channels in module
            nModuleChannels = 128; 
%             nModuleChannels = usSystem.nChTotal./usSystem.nArius; 
            
            % number of available rx channels in single module            
            nRxChannels = 32; 
%             nRxChannels = usSystem.nChArius;
            
            % number of rx channel groups
            nRxChanGroups = 3; 
            
            nElements = length(txAp);
            
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

            % alternative approach
            
%             m2e = false(nModuleChannels, nElements, nModule);
%             
%             for iModule = 1:nModules
%                 for iElement = 1:nElement
%                     for iChannel = 1:nModuleChannels
%                         
%                     end
%                 end
%             end
            
%             module2elementArray
            
            

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