classdef TxRxKernel
    
    
    properties
        
        
    end
    
    methods
        function obj = TxRxKernel(sys, TxRxSequence)
            obj.sys = sys;
            obj.TxRxSequence = TxRxSequence;
            
            
        end
        
        function programHW(obj)
            
            % Program mappings, gains, and voltage
            for iArius = 0:obj.sys.nArius-1
                
                % Set Rx channel mapping
                for iChannel = 1:32
                    Us4MEX(iArius, "SetRxChannelMapping", ...
                        obj.sys.rxChannelMap(iArius+1, iChannel), ...
                        iChannel);
                end

                % Set Tx channel mapping
                for iChannel = 1:128
                    Us4MEX(iArius, "SetTxChannelMapping", ...
                        obj.sys.txChannelMap(iArius+1, iChannel), ...
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
                    Us4MEX(0, "SetHVVoltage", obj.sys.voltage);
                    
                catch
                    warning('1st "SetHVVoltage" failed');
                    Us4MEX(0, "SetHVVoltage", obj.sys.voltage);
                    
                end
            end
            
            % Program Tx/Rx sequence
            for iArius=0:(obj.sys.nArius-1)
                for iFire=0:(obj.seq.nFire-1)
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
                    Us4MEX(iArius, "TGCSetSamples", obj.seq.tgcCurve, iFire);
                end
                Us4MEX(iArius, "SetNumberOfFirings", obj.seq.nFire);
                Us4MEX(iArius, "EnableTransmit");
                Us4MEX(iArius, "EnableReceive");
            end
            
            % Program triggering
            Us4MEX(0, "SetNTriggers", obj.seq.nTrig);
            for iTrig=0:(obj.seq.nTrig-1)
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
        
        
    end
   
end