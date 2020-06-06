
% path to the MATLAB API files
addpath('../arrus');


% parameters

nUs4OEM     = 2;
probeName	= 'SL1543';
txFrequency = 7e6;
samplingFrequency = 65e6;
fsDivider = 2;
[filtB,filtA] = butter(2,[0.5 1.5]*txFrequency/(samplingFrequency/fsDivider/2),'bandpass');

%% define sequences
sequences = cell(1);
n = 0;

% simple STA sequence
n = n+1;
sequences{n} = STASequence('txApertureCenter', (-15:3:15)*1e-3, ...
                        'txApertureSize',   32, ...
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


%% main loop

for iSeq = 1:length(sequences)
    thisSequence = sequences{iSeq};
    % us = Us4R(nUs4OEM, probeName, 50, true);
    % us.upload(thisSequence);
    % [rf,img] = us.run;
    
    
end
                    
                    
%% test functions definitions

