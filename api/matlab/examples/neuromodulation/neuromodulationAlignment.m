% Neuromodulation: alignment script
    
%% Predefined parameters
tStim = 1;
tPause = 2;

txFreq = 6.5e6;
txFoc = 20e-3;
dutyCycle = 0.5;

nTx = 100;

volt = 3;
prf = 1000;
nRep = 10;

pri = 1./prf;
nCyc = pri * dutyCycle * txFreq;

%% Initialize the system
addpath('C:\Users\pkarwat\Documents\GitHub\arrus\install\matlab');
addpath('C:\Users\pkarwat\Documents\GitHub\arrus\install\matlab\arrus\mexcuda');
addpath('../../');
addpath('../../arrus');

% Make sure the configuration in the *.prototxt file is correct.
us  = Us4R.create('configFile', 'us4r_L9-4.prototxt');

us.setMaximumPulseLength(1050e-6);

%% Program Tx/Rx sequence and reconstruction
seq = CustomTxRxSequence(   'txApertureCenter', 0e-3, ...
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
                            'txVoltage',        volt + [0, 0; 1, 1], ...  
                            'txVoltageId',      1, ...
                            'txNPeriods',       [nCyc*ones(1,nTx-1), 2], ...
                            'txPri',            pri ...
                            );

us.upload(seq);

%% Continuous in-loop operation
disp("Press any key to start...");
pause;

neuroTimer = NeuroTimer(tStim);
for iRep=1:nRep
    
    beep();
    neuroTimer.reset();
    us.runLoop(@neuroTimer.isContinue, ...
               @neuroTimer.doNothing, ...
               'bufferType', 'none');

    pause(tPause);
    
end

pause(1); beep();
pause(1); beep();
pause(1); beep();
