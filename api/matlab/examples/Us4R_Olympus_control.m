
% path to the MATLAB API files
% addpath('../arrus');
txFrequency = 5e6;
samplingFrequency = 65e6;
fsDivider = 1;

[filtB,filtA] = butter(2,[0.5 1.5]*txFrequency/(samplingFrequency/fsDivider/2),'bandpass');

%% Initialize the system, sequence, and reconstruction
nUs4OEM = 2;
us	= Us4R('nUs4OEM',      nUs4OEM, ...
           'probeName',   '5L128', ...
           'adapterType', 'esaote3', ...
           'voltage',      50, ...
           'logTime',      true);

seqSTA = STASequence(	'txApertureCenter', (-30:5:30)*1e-3, ...
                        'txApertureSize',   32, ...
                        'rxApertureCenter', 0*1e-3, ...
                        'rxApertureSize',   128, ...
                        'txFocus',          -20*1e-3, ...
                        'txAngle',          0*pi/180, ...
                        'speedOfSound',     5900, ...
                        'txFrequency',      txFrequency, ...
                        'txNPeriods',       2, ...
                        'rxNSamples',       round(4*1024/fsDivider), ...
                        'nRepetitions',     1, ...
                        'txPri',            200*1e-6, ...
                        'tgcStart',         14, ...
                        'tgcSlope',         0e2, ...
                        'fsDivider',        fsDivider);                        


seqPWI = PWISequence(	'txApertureCenter', 0*1e-3, ...
                        'txApertureSize',   128, ...
                        'rxApertureCenter', 0*1e-3, ...
                        'rxApertureSize',   128, ...
                        'txFocus',          inf*1e-3, ...
                        'txAngle',          [-25:5:25]*pi/180, ...
                        'speedOfSound',     5900, ...
                        'txFrequency',      txFrequency, ...
                        'txNPeriods',       2, ...
                        'rxNSamples',       round(4*1024/fsDivider), ...
                        'nRepetitions',     1, ...
                        'txPri',            200*1e-6, ...
                        'tgcStart',         14, ...
                        'tgcSlope',         0e2, ...
                        'fsDivider',        fsDivider);

seqLIN = LINSequence(	'txCenterElement',	1:128, ...
                        'txApertureSize',   32, ...
                        'rxCenterElement',	1:128, ...
                        'rxApertureSize',   64, ...
                        'txFocus',          40*1e-3, ...
                        'txAngle',          0*pi/180, ...
                        'speedOfSound',     5900, ...
                        'txFrequency',      txFrequency, ...
                        'txNPeriods',       2, ...
                        'rxNSamples',       round(4*1024/fsDivider), ...
                        'nRepetitions',     1, ...
                        'txPri',            200*1e-6, ...
                        'tgcStart',         14, ...
                        'tgcSlope',         0e2, ...
                        'fsDivider',        fsDivider);                        

% GPU/CPU reconstruction implemented in matlab.
rec = Reconstruction(   'filterEnable',     true, ...
                        'filterACoeff',     filtA, ...
                        'filterBCoeff',     filtB, ...
                        'iqEnable',         true, ...
                        'cicOrder',         2, ...
                        'decimation',       4, ...
                        'xGrid',            (-40:0.20:40)*1e-3, ...
                        'zGrid',            (  0:0.20:110)*1e-3);

% us.upload(seqSTA, rec);
us.upload(seqPWI, rec);
% us.upload(seqLIN, rec);

%% Run sequence and reconstruction
[rf,img] = us.run;
% 
% display = BModeDisplay((-20:0.10:20)*1e-3, (  0:0.10:50)*1e-3);
% us.runLoop(@display.isOpen, @display.updateImg);


%%
figure
    imagesc(img)
    colormap(gray)
    set(gca, 'clim', [0,100])
