
% path to the MATLAB API files
addpath('../arrus');

nUs4OEM     = 2;
probeName	= 'SL1543';
adapterType = 'esaote2';

txFrequency = 7e6;
samplingFrequency = 65e6;
fsDivider = 1;

[filtB,filtA] = butter(2,[0.5 1.5]*txFrequency/(samplingFrequency/fsDivider/2),'bandpass');

%% Initialize the system, sequence, and reconstruction
us	= Us4R(nUs4OEM, probeName, adapterType, 10, true);

%%

pulse = TxPulse('nPeriods',[2], 'frequency', [5e6]);
t1 = Tx('pulse', pulse, 'aperture', true(1,192));
r1 = Rx('aperture', true(1,192), 'time', 50e-6);
txrx1 = TxRx('Tx',t1,'Rx', r1);
sequence = TxRxSequence([txrx1,txrx1]);
%%
us.upload(sequence);
% 

%% Run sequence (no reconstruction)
[rf] = us.run;



