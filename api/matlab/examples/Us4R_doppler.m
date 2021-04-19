
% path to the MATLAB API files
addpath('../arrus');

txFrequency = 18e6;
samplingFrequency = 65e6;
fsDivider = 1;

[filtB,filtA] = butter(2,[0.5 1.5]*txFrequency/(samplingFrequency/fsDivider/2),'bandpass');
[wcfB, wcfA]  = butter(2,0.1,'high');

nUs4OEM = 2;
%% Initialize the system, sequence, and reconstruction
us	= Us4R('nUs4OEM',      nUs4OEM, ...
           'probeName',   'LA/20/128', ...
           'adapterType', 'atl/philips', ...
           'voltage',      15, ...
           'logTime',      true);

seqSTA = STASequence(	'txApertureCenter', (-15:3:15)*1e-3, ...
                        'txApertureSize',   32, ...
                        'rxApertureCenter', 0*1e-3, ...
                        'rxApertureSize',   192, ...
                        'txFocus',          -6*1e-3, ...
                        'txAngle',          0*pi/180, ...
                        'speedOfSound',     1450, ...
                        'txFrequency',      txFrequency, ...
                        'txNPeriods',       2, ...
                        'rxNSamples',       round(8*1024/fsDivider), ...
                        'nRepetitions',     1, ...
                        'txPri',            200*1e-6, ...
                        'tgcStart',         14, ...
                        'tgcSlope',         2e2, ...
                        'fsDivider',        fsDivider);                        


seqPWI = PWISequence(	'txApertureCenter', 0*1e-3, ...
                        'txApertureSize',   128, ...
                        'rxApertureCenter', 0*1e-3, ...
                        'rxApertureSize',   128, ...
                        'txFocus',          inf*1e-3, ...
                        'txAngle',          [00*ones(1,64)]*pi/180, ...
                        'speedOfSound',     1490, ...
                        'txFrequency',      txFrequency, ...
                        'txNPeriods',       4, ...
                        'rxDepthRange',     [0e-3,20e-3], ...                 
                        'nRepetitions',     1, ...
                        'txPri',            200*1e-6, ...
                        'tgcStart',         14, ...
                        'tgcSlope',         6e2, ...
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
                        'rxNSamples',       round(8*1024/fsDivider), ...
                        'nRepetitions',     1, ...
                        'txPri',            200*1e-6, ...
                        'tgcStart',         14, ...
                        'tgcSlope',         2e2, ...
                        'txInvert',         0, ...
                        'fsDivider',        fsDivider);                        

% GPU/CPU reconstruction implemented in matlab.
rec = Reconstruction(   'filterEnable',     true, ...
                        'filterACoeff',     filtA, ...
                        'filterBCoeff',     filtB, ...
                        'iqEnable',         true, ...
                        'cicOrder',         2, ...
                        'decimation',       2, ...
                        'xGrid',            (-7:0.025:7)*1e-3, ...
                        'zGrid',            (  0:0.025:20)*1e-3, ...
                        'dopplerEnable',	true, ...
                        'vectorEnable',     true, ...
                        'wcFilterBCoeff',   wcfB, ...
                        'wcFilterACoeff',	wcfA);

% us.upload(seqSTA, rec);
us.upload(seqPWI, rec);
% us.upload(seqLIN, rec);

%% Run sequence and reconstruction
% [rf,img] = us.run;
% 
display = DuplexDisplay((-7:0.025:7)*1e-3, (  0:0.025:20)*1e-3, [0 60], 20);
us.runLoop(@display.isOpen, @display.updateImg);