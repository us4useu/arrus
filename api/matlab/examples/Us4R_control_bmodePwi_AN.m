% Script for PWI measurement for AN

% Use SL1543 probe with esaote3 adapter
% Make sure a proper prototxt is indicated
% Restart (also unplug/plug the power supply cable) the Us4R-lite before the measurement !!!

%% paths to the MATLAB ARRUS files
addpath('..\');
addpath('..\arrus');

%% Initialize the system
us = Us4R(  'configFile',   'us4r.prototxt');

nRep = 200;

seq = CustomTxRxSequence(   'txApertureCenter', 0*1e-3, ...
                            'txApertureSize',   us.getNProbeElem, ...
                            'rxApertureCenter', 0*1e-3, ...
                            'rxApertureSize',   us.getNProbeElem, ...
                            'txFocus',          inf*1e-3, ...
                            'txAngle',          [0 0]*pi/180, ...   % (problems for single TX)
                            'speedOfSound',     1490, ...
                            'txFrequency',      6e6, ...
                            'txNPeriods',       2, ...
                            'txVoltage',        80, ...
                            'rxDepthRange',     [0e-3,70e-3], ...
                            'txPri',            400e-6, ...
                            'nRepetitions',     nRep);

% GPU/CPU reconstruction implemented in matlab.
rec = Reconstruction(   'xGrid',            (-25:0.10:25)*1e-3, ...
                        'zGrid',            (  0:0.10:60)*1e-3);

us.upload(seq, rec);

%% Run
% Preview - continuous in-loop operation
display = BModeDisplay(rec, 'dynamicRange', [0 80]);
us.runLoop(@display.isOpen, @display.updateImg);

% Measurement
us.disableReconstruct();
us.reduceVoltage(10);
iq = us.run;

%% Close session
us.closeSession;

%% Offline reconstruction
img = cell(1,1,nRep);
for iRep=1:nRep
    img{iRep} = us.reconstructOffline(iq(:,:,:,iRep));
end
img = cell2mat(img);


