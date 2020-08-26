
% path to the MATLAB API filesqui
% addpath('../arrus');

nUs4OEM     = 2;


adapterType = 'esaote2';


probeName	= 'SL1543';
txFrequency = 7e6;


samplingFrequency = 65e6;
fsDivider = 1;

[filtB,filtA] = butter(2,[0.5 1.5]*txFrequency/(samplingFrequency/fsDivider/2),'bandpass');
% restart


%% Initialize the system, sequence, and reconstruction
us	= Us4R(nUs4OEM, probeName, adapterType, 1, true);

%%
% 
% txap = false(1,192);
% txap(96) = true;
% TODO: pri powinno si? dostosowywa? do liczby sampli i delay. 
%   dorobic Rx.nSamp i Rx.startSamp
%   Tx('pulse') -> Tx('excitation')?



txap = false(1,32); txap(1:32) = true;

% rxap = true(1,192);
rxap = false(1,32); rxap(1:32) = true; 

pulse = Pulse('nPeriods',[2], 'frequency', [7e6]);
t1 = Tx('pulse', pulse, 'aperture', txap);
% r1 = Rx('aperture', rxap, 'time', 7e-6,'delay',15e-6);
r1 = Rx('aperture', rxap, 'time', 50e-6, 'delay', 10e-6);
txrx1 = TxRx('Tx',t1,'Rx', r1, 'pri', 150e-6);
sequence = TxRxSequence([txrx1]);
%%
tic
us.upload(sequence);
% 

%% Run sequence (no reconstruction)
[rf] = us.run;

toc

%% images
% for i = 1:size(rf,3)
%     figure, imagesc(log(double(rf(:,:,i)).^2+1))
% end

figure, imagesc(log(double(rf(:,:,1)).^2+1))
% set(gca,'xlim',[45,75])
% set(gca,'ylim',[1300,1900])

figure, 
    imagesc(rf(900:1400,:,1))