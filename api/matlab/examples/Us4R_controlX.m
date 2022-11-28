% path to the MATLAB API files
addpath('../arrus');
addpath("C:\Users\pkarwat\Documents\GitHub\arrus\install\matlab");
addpath("C:\Users\pkarwat\Documents\GitHub\arrus\api\matlab");

%% Params
fs = 65e6;
fn = 6.5e6;
dec = 10;

%% OLD
us	= Us4R_old( 'probeName',   'SL1543', ...
                'adapterType', 'esaote3', ...
                'voltage',      10, ...
                'logTime',      true);

seq = CustomTxRxSequence(  'txApertureCenter',  0*1e-3, ...
                            'txApertureSize',   192, ...
                            'rxApertureCenter', 0*1e-3, ...
                            'rxApertureSize',   192, ...
                            'txFocus',          inf*1e-3, ...
                            'txAngle',          [-10 0 10]*pi/180, ...
                            'speedOfSound',     1490, ...
                            'txFrequency',      fn, ...
                            'txNPeriods',       2, ...
                            'rxNSamples',       3*1024, ...
                            'tgcStart',         14, ...
                            'tgcSlope',         2e2, ...
                            'txPri',            500*1e-6, ...
                            'iqEnable',         false, ...
                            'decimation',       1);

rec = Reconstruction(       'filterEnable',     false, ...
                            'iqEnable',         true, ...
                            'decimation',       10, ...
                            'xGrid',            (-20:0.10:20)*1e-3, ...
                            'zGrid',            (  0:0.10:30)*1e-3);

us.upload(seq,rec);
[rf0,img0] = us.run;
rf0 = double(rf0);

munlock('Us4MEX'); clear Us4MEX;

%% DDC
t = (0:size(rf0,1)-1)'/fs;
iq0 = 2*rf0.*exp(-2i*pi*reshape(fn,1,1,[]).*t);

% for ord=1:2
%     iq0 = cumsum(iq0);
% end
% iq0 = iq0(dec:dec:end,:,:);
% for ord=1:2
%     iq0 = [iq0(1,:,:); diff(iq0)];
% end
% iq0 = iq0 / (dec.^2);

ddcFir = ones(2*round(fs/fn),1);
ddcFir = ddcFir/sum(ddcFir);
iq0 = convn(iq0,ddcFir,'same');
iq0 = iq0(dec:dec:end,:,:);

%% NEW
us	= Us4R(     'probeName',   'SL1543', ...
                'adapterType', 'esaote3', ...
                'voltage',      10, ...
                'logTime',      true);

seq = CustomTxRxSequence(  'txApertureCenter',  0*1e-3, ...
                            'txApertureSize',   192, ...
                            'rxApertureCenter', 0*1e-3, ...
                            'rxApertureSize',   192, ...
                            'txFocus',          inf*1e-3, ...
                            'txAngle',          [-10 0 10]*pi/180, ...
                            'speedOfSound',     1490, ...
                            'txFrequency',      fn, ...
                            'txNPeriods',       2, ...
                            'rxNSamples',       round(3*1024 / dec / 64)*64, ...
                            'tgcStart',         14, ...
                            'tgcSlope',         2e2, ...
                            'txPri',            500*1e-6, ...
                            'iqEnable',         true, ...
                            'decimation',       dec);

rec = Reconstruction(       'filterEnable',     false, ...
                            'iqEnable',         false, ...
                            'decimation',       1, ...
                            'xGrid',            (-20:0.10:20)*1e-3, ...
                            'zGrid',            (  0:0.10:30)*1e-3);

us.upload(seq,rec);
[iq1,img1] = us.run;
iq1 = double(iq1);

%% Display
sLim = [850 1050]/dec;
xLim = [120 260];
zLim = [60 120];

aLim0 = 2e3;
aLim1 = aLim0 * 10;
bLim0 = [10 70];
bLim1 = bLim0 + 20;

figure;

subplot(4,2,1), imagesc(real(iq0(:,:,2))), colormap(gca,"jet"), colorbar, set(gca,'CLim',[-1 1]*aLim0,'YLim',sLim);
subplot(4,2,2), imagesc(real(iq1(:,:,2))), colormap(gca,"jet"), colorbar, set(gca,'CLim',[-1 1]*aLim1,'YLim',sLim);

subplot(4,2,3), imagesc( abs(iq0(:,:,2))), colormap(gca,"jet"), colorbar, set(gca,'CLim',[-1 1]*aLim0,'YLim',sLim);
subplot(4,2,4), imagesc( abs(iq1(:,:,2))), colormap(gca,"jet"), colorbar, set(gca,'CLim',[-1 1]*aLim1,'YLim',sLim);

subplot(4,2,5), imagesc(angle(iq0(:,:,2))), colormap(gca,"jet"), colorbar, set(gca,'CLim',[-1 1]*pi,'YLim',sLim);
subplot(4,2,6), imagesc(angle(iq1(:,:,2))), colormap(gca,"jet"), colorbar, set(gca,'CLim',[-1 1]*pi,'YLim',sLim);

subplot(4,2,7), imagesc(img0), colormap(gca,"gray"), colorbar, set(gca,'CLim',bLim0,'XLim',xLim,'YLim',zLim), daspect([1 1 1]);
subplot(4,2,8), imagesc(img1), colormap(gca,"gray"), colorbar, set(gca,'CLim',bLim1,'XLim',xLim,'YLim',zLim), daspect([1 1 1]);

