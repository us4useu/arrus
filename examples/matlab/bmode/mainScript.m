
%% --------------------------------------------------------
%% ------------------------ SETUP -------------------------
%% --------------------------------------------------------

% ---- Phantom parameters ---- (phantom = sparse rectangular grid of point scatterers)
phantomParameters.speedOfSound = 1540;              % [m/s] speed of sound
phantomParameters.attenuation = 0.5e-4;             % [dB/m/Hz] linear attenuation coefficient

xGrid       = (-15:5:15)*1e-3;                      % [m] x-grid vector of point sources
yGrid       = 0*1e-3;                               % [m] y-grid vector of point sources
zGrid       = (10:5:40)*1e-3;                       % [m] z-grid vector of point sources

[phantomParameters.particles.xPosition, ...         % [m] x-position of point sources
phantomParameters.particles.zPosition, ...
phantomParameters.particles.yPosition] = meshgrid(xGrid,zGrid,yGrid);
phantomParameters.particles.scattering = ones(size(phantomParameters.particles.xPosition));

% ---- System parameters ----
systemParameters.nElements = 192;                   % [elem] number of probe elements
systemParameters.pitch = 0.21e-3;                   % [m] transducer pitch

% ---- Acquisition parameters ----
acquisitionParameters.mode = 'lin';

acquisitionParameters.speedOfSound = 1540;          % [m/s] speed of sound
acquisitionParameters.attenuation = 0.5e-4;         % [dB/m/Hz] linear attenuation coefficient

acquisitionParameters.tx.frequency = 5e6;           % [Hz] carrier frequency
acquisitionParameters.tx.nPeriods = 2;              % [] number of sine periods in tx burst
acquisitionParameters.tx.angle = 0*(pi/180);        % [rad] tx angles (for PWI,LIN)
acquisitionParameters.tx.focus = 20e-3;             % [m] tx focus (for LIN)
acquisitionParameters.tx.apertureSize = 32;         % [elem] tx aperture (for LIN)

acquisitionParameters.rx.samplingFrequency = 65e6;	% [Hz] sampling frequency
acquisitionParameters.rx.apertureSize = 48;         % [elem] rx aperture (for LIN)

% ---- Processing parameters ----
% Band-pass filtration
filterOrder = 2;
cutoffLow = 0.5*acquisitionParameters.tx.frequency/(acquisitionParameters.rx.samplingFrequency/2);
cutoffHigh = 1.5*acquisitionParameters.tx.frequency/(acquisitionParameters.rx.samplingFrequency/2);
cutoffMean = (cutoffLow+cutoffHigh)/2;
[b,a] = butter(filterOrder,[cutoffLow cutoffHigh],'bandpass');
del = phasez(b,a,[0 cutoffMean],acquisitionParameters.rx.samplingFrequency) ...
    /(2*pi)/acquisitionParameters.tx.frequency*acquisitionParameters.rx.samplingFrequency;
del = -del(2);

processingParameters.filter.enable = true;
processingParameters.filter.b = b;
processingParameters.filter.a = a;
processingParameters.filter.delay = del;

% Digital Down Conversion
processingParameters.ddc.iqEnable = true;
processingParameters.ddc.cicOrder = 2;
processingParameters.ddc.decimation = 2;

% Delay And Sum
processingParameters.das.xGrid = (-20:0.1:20)*1e-3;	% [m]
processingParameters.das.zGrid = (0:0.05:50)*1e-3; % [m]

%% --------------------------------------------------------
%% -------------------- RF SIMULATION ---------------------
%% --------------------------------------------------------

% add path to Field II (this one works for me)
addpath([userpath '\Lib_Field']);

% simulate rf data for SSTA scheme
rfRaw = simulateRfSta(systemParameters,acquisitionParameters,phantomParameters);

% convert the SSTA data to LIN or PWI format
if any(strcmp(acquisitionParameters.mode,{'pwi','lin'}))
    rfRaw = convertRfSta(rfRaw,systemParameters,acquisitionParameters);
end

%% --------------------------------------------------------
%% ---------------------- PROCESSING ----------------------
%% --------------------------------------------------------

%% Move data to GPU if possible
gpuEnable	= license('test', 'Distrib_Computing_Toolbox') && ~isempty(ver('distcomp'));
if gpuEnable
    rfRaw = gpuArray(rfRaw);
end

%% Preprocessing

% Raw rf data filtration
if processingParameters.filter.enable
    rfRaw = filter(processingParameters.filter.b,processingParameters.filter.a,rfRaw);
end

% Digital Down Conversion
rfRaw = downConv(rfRaw,acquisitionParameters,processingParameters);

% warning: both filtration and decimation introduce phase delay!

%% Image reconstruction
if strcmp(acquisitionParameters.mode,'lin')
    % tx delay of aperture center - far from ideal (does not account for acquisitionParameters.tx.angle and tx aperture clipping)
    txCentDel= (sqrt(acquisitionParameters.tx.focus^2 + (acquisitionParameters.tx.apertureSize/2*systemParameters.pitch).^2) - acquisitionParameters.tx.focus)/acquisitionParameters.speedOfSound;
    rfBfr = reconstructRfLin(rfRaw,systemParameters,acquisitionParameters,processingParameters,txCentDel);
else
    rfBfr = reconstructRfImg(rfRaw,systemParameters,acquisitionParameters,processingParameters);
end

%% --------------------------------------------------------
%% ------------------- POSTPROCESSING ---------------------
%% --------------------------------------------------------
% Obtain complex signal (if it isn't complex already)
if ~processingParameters.ddc.iqEnable
    nanMask = isnan(rfBfr);
    rfBfr(nanMask) = 0;
    rfBfr = hilbert(rfBfr);
    rfBfr(nanMask) = nan;
end

% Scan conversion (for 'lin' mode)
if strcmp(acquisitionParameters.mode,'lin')
    rfBfr = scanConvert(rfBfr,systemParameters,acquisitionParameters,processingParameters);
end

% Envelope detection
envImg = abs(rfBfr);

% Compression
imgBMode = 20*log10(envImg);

%% --------------------------------------------------------
%% ----------------------- DISPLAY ------------------------
%% --------------------------------------------------------
if gpuEnable
    imgBMode = gather(imgBMode);
end

dynRng	= 40;
cMax = max(imgBMode(:));

figure;
imagesc(processingParameters.das.xGrid*1e3,processingParameters.das.zGrid*1e3,imgBMode);
set(gca,'CLim',cMax + [-dynRng 0]);
daspect([1 1 1]);
colormap(gray);
colorbar;
xlabel('x [mm]');
ylabel('z [mm]');



