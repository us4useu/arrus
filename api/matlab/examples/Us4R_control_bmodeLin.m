
%% paths to the MATLAB API/ARRUS files
pathArrus = 'C:\Users\user\Documents\GitHub\arrus\'; % path to the arrus repo

addpath([pathArrus 'install\matlab']);
addpath([pathArrus 'api\matlab']);
addpath([pathArrus 'api\matlab\arrus']);

%% Initialize the system
us = Us4R(  'configFile',   'us4r.prototxt', ...
            'voltage',      20);

seq = CustomTxRxSequence(   'txCenterElement',	1:us.getNProbeElem, ...
                            'txApertureSize',   32, ...
                            'rxCenterElement',	1:us.getNProbeElem, ...
                            'rxApertureSize',   32, ...
                            'txFocus',          20*1e-3, ...
                            'txAngle',          0*pi/180, ...
                            'speedOfSound',     1490, ...
                            'txFrequency',      6e6, ...
                            'txNPeriods',       2, ...
                            'rxDepthRange',     [0e-3,50e-3]);

% GPU/CPU reconstruction implemented in matlab.
rec = Reconstruction(   'xGrid',            (-20:0.10:20)*1e-3, ...
                        'zGrid',            (  0:0.10:50)*1e-3, ...
                        'gridModeEnable',   false);

us.upload(seq, rec);

%% Run sequence and reconstruction
% Single execution of the sequence, collecting iq & img data
[iq,img] = us.run;

% Continuous in-loop operation
display = BModeDisplay(rec, 'dynamicRange', [0 80]);
us.runLoop(@display.isOpen, @display.updateImg);

%% Close session
us.closeSession;


