
% path to the MATLAB API filesqui
addpath('../arrus');

nUs4OEM     = 2;
probeName	= 'SL1543';
% probeName = 'sp2430';
adapterType = 'esaote2';

txFrequency = 7e6;
samplingFrequency = 65e6;
fsDivider = 1;

[filtB,filtA] = butter(2,[0.5 1.5]*txFrequency/(samplingFrequency/fsDivider/2),'bandpass');
% restart


%% Initialize the system, sequence, and reconstruction
us	= Us4R(nUs4OEM, probeName, adapterType, 50, true);

%%
% 
% txap = false(1,192);
% txap(96) = true;

txap = true(1,192);
rxap = true(1,192);

pulse = Pulse('nPeriods',[2], 'frequency', [5e6]);
t1 = Tx('pulse', pulse, 'aperture', txap);
r1 = Rx('aperture', rxap, 'time', 7e-6,'delay',15e-6);
txrx1 = TxRx('Tx',t1,'Rx', r1);
sequence = TxRxSequence([txrx1]);
%%
tic
us.upload(sequence);
% 

%% Run sequence (no reconstruction)
[rf] = us.run;

toc

%%
figure, imagesc(rf(:,:,1).^1)
% set(gca, 'ylim', [4980, 5100])
