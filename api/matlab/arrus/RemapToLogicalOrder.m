classdef RemapToLogicalOrder
    % Remaps from system physical order to the probe-like logical order.

    properties
        nRx
        nTx
        nRep
        hwDdcEnable
        reorgMap
    end

    methods
        function obj = RemapToLogicalOrder(scheme, metadata)
            
            addpath([fileparts(mfilename('fullpath')) '\mexcuda']);
            
            framesOffset = metadata{1};
            framesNumber = metadata{2};
            oemId = metadata{3};
            frameId = metadata{4};
            channelId = metadata{5};
            
            nRx = sum(scheme.txRxSequence.ops(1).rx.aperture);
            nTx = length(scheme.txRxSequence.ops);
            nRep = scheme.txRxSequence.nRepeats;
            nOem = numel(framesNumber);
            nChunk = sum(framesNumber);
            nChan = 32;
            
            obj.nRx = uint32(nRx);
            obj.nTx = uint32(nTx);
            obj.nRep = uint32(nRep);
            obj.hwDdcEnable = ~isempty(scheme.digitalDownConversion);
            
            obj.reorgMap = - ones(nChan, nChunk, 'int32');
            
            for iOem=1:nOem
                nFrame = framesNumber(iOem) / nRep;
                for iFrame=1:nFrame
                    isSelect = oemId == iOem-1 ...
                             & frameId == iFrame-1 ...
                             & channelId >= 0;
                    iTx = find(any(isSelect));
                    iRx = find(isSelect(:,iTx) & channelId(:,iTx) >= 0);
                    iChan = channelId(iRx,iTx) + 1;
                    for iRep=1:nRep
                        iChunk = framesOffset(iOem) ...
                               + framesNumber(iOem) / nRep * (iRep-1) ...
                               + iFrame;
                        
                        % 0-based indexing
                        obj.reorgMap(iChan,iChunk) = (iRep-1)*nTx*nRx + (iTx-1)*nRx + iRx-1; 
                    end
                end
            end
        end

        function dataOut = process(obj, dataIn)
            dataIn = gpuArray(dataIn);

            dataOut = rawReorg( dataIn, ...
                                obj.reorgMap, ...
                                obj.nRx, ...
                                obj.nTx, ...
                                obj.nRep, ...
                                obj.hwDdcEnable);
        end
    end
end