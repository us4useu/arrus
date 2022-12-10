
%% paths to the MATLAB ARRUS files
addpath('..\');
addpath('..\arrus');

%% Initialize the system
us = Us4R(  'configFile',   'us4r.prototxt');

seq = CustomTxRxSequence(   'txApertureCenter', 0*1e-3, ...
                            'txApertureSize',   us.getNProbeElem, ...
                            'rxApertureCenter', 0*1e-3, ...
                            'rxApertureSize',   us.getNProbeElem, ...
                            'txFocus',          inf*1e-3, ...
                            'txAngle',          (-15:3:15)*pi/180, ...
                            'speedOfSound',     1490, ...
                            'txFrequency',      6e6, ...
                            'txNPeriods',       2, ...
                            'txVoltage',        20, ...
                            'rxDepthRange',     [0e-3,50e-3]);

% GPU/CPU reconstruction implemented in matlab.
rec = Reconstruction(   'xGrid',            (-20:0.10:20)*1e-3, ...
                        'zGrid',            (  0:0.10:50)*1e-3);

us.upload(seq, rec);

%% Run sequence and reconstruction
% Single execution of the sequence, collecting iq & img data
[iq,img] = us.run;

% Continuous in-loop operation
display = BModeDisplay(rec, 'dynamicRange', [0 80]);
us.runLoop(@display.isOpen, @display.updateImg);

%% Close session
us.closeSession;


