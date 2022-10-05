
% path to the MATLAB API files
addpath('../arrus');
addpath("C:\Users\pkarwat\Documents\GitHub\arrus\install\matlab");
addpath("C:\Users\pkarwat\Documents\GitHub\arrus\api\matlab");

%% Initialize the system, sequence, and reconstruction
if 1
    us	= Us4R_old( 'probeName',   'SL1543', ...
                    'adapterType', 'esaote3', ...
                    'voltage',      50, ...
                    'logTime',      true);
else
    us	= Us4R(     'probeName',   'SL1543', ...
                    'adapterType', 'esaote3', ...
                    'voltage',      50, ...
                    'logTime',      true);
end

seq1 = CustomTxRxSequence(  'txApertureCenter', 0*1e-3, ...
                            'txApertureSize',   [192 192 32 64], ...
                            'rxApertureCenter', 0*1e-3, ...
                            'rxApertureSize',   140, ...
                            'txFocus',          [inf inf -6 20]*1e-3, ...
                            'txAngle',          [0 5 0 0]*pi/180, ...
                            'speedOfSound',     1490, ...
                            'txFrequency',      6e6, ...
                            'txNPeriods',       2, ...
                            'rxNSamples',       3*1024, ...
                            'tgcStart',         14, ...
                            'tgcSlope',         2e2, ...
                            'txPri',            500*1e-6);

seq2 = CustomTxRxSequence(  'txCenterElement',	90:2:110, ...
                            'txApertureSize',   32, ...
                            'rxCenterElement',	90:2:110, ...
                            'rxApertureSize',   40, ...
                            'txFocus',          30*1e-3, ...
                            'txAngle',          0*pi/180, ...
                            'speedOfSound',     1490, ...
                            'txFrequency',      6e6, ...
                            'txNPeriods',       2, ...
                            'rxNSamples',       3*1024, ...
                            'tgcStart',         14, ...
                            'tgcSlope',         2e2, ...
                            'txPri',            500*1e-6);

us.upload(seq1);
rf1 = us.run;

us.upload(seq2);
rf2 = us.run;

%%
figure;
subplot(2,1,1), imagesc(rf1(:,:)), colormap("jet"), colorbar, set(gca,'CLim',[-1 1]*1e4,'YLim',[1400 2000]);
subplot(2,1,2), imagesc(rf2(:,:)), colormap("jet"), colorbar, set(gca,'CLim',[-1 1]*1e4,'YLim',[1400 2000]);

