
% path to the MATLAB API files
addpath('../arrus');

nUs4OEM     = 2;
probeName	= 'SL1543';
adapterType = 'esaotev2';

txFrequency = 7e6;
samplingFrequency = 65e6;
fsDivider = 1;

[filtB,filtA] = butter(2,[0.5 1.5]*txFrequency/(samplingFrequency/fsDivider/2),'bandpass');

%% Initialize the system, sequence, and reconstruction
us	= Us4R(nUs4OEM, probeName, adapterType, 50, true);

seqSTA = STASequence(	'txApertureCenter', (-15:3:15)*1e-3, ...
                        'txApertureSize',   32, ...
                        'rxApertureCenter', 0*1e-3, ...
                        'rxApertureSize',   192, ...
                        'txFocus',          -6*1e-3, ...
                        'txAngle',          0*pi/180, ...
                        'speedOfSound',     1450, ...
                        'txFrequency',      txFrequency, ...
                        'txNPeriods',       2, ...
                        'rxNSamples',       8*1024/fsDivider, ...
                        'nRepetitions',     1, ...
                        'txPri',            200*1e-6, ...
                        'tgcStart',         14, ...
                        'tgcSlope',         2e2, ...
                        'fsDivider',        fsDivider);                        


seqPWI = PWISequence(	'txApertureCenter', 0*1e-3, ...
                        'txApertureSize',   192, ...
                        'rxApertureCenter', 0*1e-3, ...
                        'rxApertureSize',   192, ...
                        'txFocus',          inf*1e-3, ...
                        'txAngle',          [0]*pi/180, ...
                        'speedOfSound',     1450, ...
                        'txFrequency',      txFrequency, ...
                        'txNPeriods',       2, ...
                        'rxDepthRange',     [0e-3,40e-3], ...                 
                        'nRepetitions',     1, ...
                        'txPri',            200*1e-6, ...
                        'tgcStart',         14, ...
                        'tgcSlope',         2e2, ...
                        'fsDivider',        fsDivider);

seqLIN = LINSequence(	'txCenterElement',	1:192, ...
                        'txApertureSize',   32, ...
                        'rxCenterElement',	1:192, ...
                        'rxApertureSize',   32, ...
                        'txFocus',          20*1e-3, ...
                        'txAngle',          0*pi/180, ...
                        'speedOfSound',     1450, ...
                        'txFrequency',      txFrequency, ...
                        'txNPeriods',       2, ...
                        'rxNSamples',       8*1024/fsDivider, ...
                        'nRepetitions',     1, ...
                        'txPri',            200*1e-6, ...
                        'tgcStart',         14, ...
                        'tgcSlope',         2e2, ...
                        'fsDivider',        fsDivider);                        

% GPU/CPU reconstruction implemented in matlab.
rec = Reconstruction(   'filterEnable',     true, ...
                        'filterACoeff',     filtA, ...
                        'filterBCoeff',     filtB, ...
                        'iqEnable',         true, ...
                        'cicOrder',         2, ...
                        'decimation',       4, ...
                        'xGrid',            (-20:0.10:20)*1e-3, ...
                        'zGrid',            (  0:0.10:50)*1e-3);

% us.upload(seqSTA,rec);
us.upload(seqPWI,rec);
% us.upload(seqLIN,rec);

%% Run sequence and reconstruction
% [rf,img] = us.run;
% 
display = BModeDisplay((-20:0.10:20)*1e-3, (  0:0.10:50)*1e-3);
us.runLoop(@display.isOpen, @display.updateImg);


