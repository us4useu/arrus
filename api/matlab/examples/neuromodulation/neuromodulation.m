% Neuromodulation function
function [sri] = neuromodulation(volt,prf,nRep,tStim,tPause)
    
    %% Predefined parameters
    % tStim = 60;
    % tPause = 60;
    
    txFreq = 6.5e6;
    txFoc = 20e-3;
    dutyCycle = 0.5;

    nTx = 100;

    %% Validation of input parameters
    if any(volt < 1) || any(volt > 3)
        error("Invalid volt value, can be 1-3");
    end

    if any(prf < 500) || any(prf > 5000)
        error("Invalid prf value, can be 500-5000");
    end

    if ~isscalar(volt) && ~isscalar(prf) && numel(volt) ~= numel(prf)
        error("Incompatible length of volt and pri");
    end

    if nargin<3
        nRep = 1;
    end

    nSeq = max(numel(volt),numel(prf));
    if nTx*nSeq*nRep > 1000
        error("Sequence is too long");
    end
    if isscalar(volt)
        volt = volt.*ones(1,nSeq);
    end
    if isscalar(prf)
        prf = prf.*ones(1,nSeq);
    end
    
    %% Initialize the system
    addpath('C:\Users\pkarwat\Documents\GitHub\arrus\install\matlab');
    addpath('C:\Users\pkarwat\Documents\GitHub\arrus\install\matlab\arrus\mexcuda');
    addpath('../../');
    addpath('../../arrus');
    
    % Make sure the configuration in the *.prototxt file is correct.
    us  = Us4R.create('configFile', 'us4r_L9-4.prototxt');

    us.setMaximumPulseLength(1050e-6);
    
    %% Program Tx/Rx sequence and reconstruction
    pri = 1./prf;
    nCyc = pri * dutyCycle * txFreq;
    
    seq = cell(1,nSeq);
    for iSeq=1:nSeq
        seq{iSeq} = CustomTxRxSequence( 'txApertureCenter', 0e-3, ...
                                        'txApertureSize',   64, ...
                                        'rxApertureCenter', 0e-3, ...
                                        'rxApertureSize',   [zeros(1,nTx-1), 64], ...
                                        'txFocus',          txFoc, ...
                                        'txAngle',          0, ...
                                        'speedOfSound',     1480, ...
                                        'txFrequency',      txFreq, ...
                                        'rxNSamples',       64, ...
                                        'hwDdcEnable',      false, ...
                                        'decimation',       1, ...
                                        'workMode',         'SYNC', ...
                                        'txVoltage',        volt(1) + [0, 0; 1, 1], ...  
                                        'txVoltageId',      1, ...
                                        'txNPeriods',       [nCyc(iSeq)*ones(1,nTx-1), 2], ...
                                        'txPri',            pri(iSeq) ...
                                        );
    end

    us.uploadSequence([seq{:}]);
    
    %% Continuous in-loop operation
    buffSize = max(ceil(tStim * prf / nTx)) + 3;
    sri = nan(buffSize,nSeq*nRep);
    
    beep();
    disp("Press any key to start the 10 sec countdown...");
    pause;
    pause(8); beep();
    pause(1); beep();
    pause(1); beep();
    
    neuroTimer = NeuroTimer(tStim);
    for iRep=1:nRep
        for iSeq=1:nSeq
            %% Select sequence and set voltage
            tic;
            us.selectSequence(iSeq);
            us.setVoltage(volt(iSeq) + [0, 0; 1, 1]);
            reloadTime = toc;
            disp(['Reload time: ' num2str(reloadTime) 's']);
    
            %% Wait
            if reloadTime > tPause
                error("Failed to load the parameters on time");
            end
            pause(tPause-reloadTime);
            
            %% Stimulate
            beep();
            neuroTimer.reset();
            [~,~,sri(:,iSeq+(iRep-1)*nSeq)] = us.runLoop(@neuroTimer.isContinue, ...
                                                         @neuroTimer.doNothing, ...
                                                         'bufferType', 'none', ...
                                                         'bufferSize', buffSize);
            beep();
        end
    end
    
    pause(1); beep();
    pause(1); beep();
    
    % figure, plot(sri), grid on;

end

