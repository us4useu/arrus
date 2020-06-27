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
            
            nArius = obj.usSystem.nArius; % number of arius modules
            nRxChannels = obj.usSystem.nChArius; % max number of rx channels 
            nTxChannels = 128; % max number of tx channels
            nFire = obj.calcNFire(); % number of firings in the sequence
            
            
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
            
            
            
            
            % tutaj zmienic
            
            
            % Program Tx/Rx sequence
            for iArius = 0:nArius-1
                for iFire = 0:nFire-1
                    
                    % active channel groups
                    Us4MEX(iArius, "SetActiveChannelGroup", obj.seq.actChanGroupMask(iArius+1), iFire);
                    
                    % Tx
                    iTx     = 1 + floor(iFire/obj.seq.nSubTx);
                    Us4MEX(iArius, "SetTxAperture", obj.seq.txSubApMask(iArius+1,iTx), iFire);
                    Us4MEX(iArius, "SetTxDelays", obj.seq.txSubApDel{iArius+1,iTx}, iFire);
                    Us4MEX(iArius, "SetTxFrequency", obj.seq.txFreq, iFire);
                    Us4MEX(iArius, "SetTxHalfPeriods", obj.seq.txNPer*2, iFire);
                    Us4MEX(iArius, "SetTxInvert", 0, iFire);
                    
                    % Rx
                    Us4MEX(iArius, "SetRxAperture", obj.seq.rxSubApMask(iArius+1,iFire+1), iFire);
                    Us4MEX(iArius, "SetRxTime", obj.seq.rxTime, iFire);
                    Us4MEX(iArius, "SetRxDelay", obj.seq.rxDel, iFire);
%                     Us4MEX(iArius, "TGCSetSamples", obj.seq.tgcCurve, iFire);
                end
                
                Us4MEX(iArius, "SetNumberOfFirings", obj.seq.nFire);
                Us4MEX(iArius, "EnableTransmit");
                Us4MEX(iArius, "EnableReceive");
            end
            
            % Program triggering
            Us4MEX(0, "SetNTriggers", nFire);
            for iTrig=0:nFire-1
                Us4MEX(0, "SetTrigger", obj.seq.txPri*1e6, 0, 0, iTrig);
            end
            Us4MEX(0, "SetTrigger", obj.seq.txPri*1e6, 0, 1, obj.seq.nTrig-1);
            for iArius=1:(obj.sys.nArius-1)
                Us4MEX(iArius, "SetTrigger", obj.seq.txPri*1e6, 0, 0, 0);
            end
            
            % Program recording
            for iArius=0:(obj.sys.nArius-1)
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
            
        end
        
        
        function nFire = calcNFire(obj)
            % The method calculate the number of firings neccessary to
            % realize the TxRxSequence
            
%             maxRxChannels = obj.sys.nChArius;
            maxRxChannels = 32;
            nFire = 0;
            for iTxRx = 1:length(obj.sequence.TxRxList)
                thisRxAperture = obj.sequence.TxRxList(1,iTxRx).Rx.rxAperture;
                apertureLenght = length(thisRxAperture);
                iTxRxFirings = ceil(apertureLenght./maxRxChannels);
                nFire = nFire + iTxRxFirings;

            end
        end
        
        
    end
   
end