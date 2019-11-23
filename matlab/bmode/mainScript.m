
%% System setup
sys.nElem = 192;            % [elem] number of transducer elements
sys.pitch = 0.21e-3;        % [m] transducer pitch
sys.fs = 65e6;              % [Hz] sampling frequency
sys.fn = 5e6;               % [Hz] carrier frequency
sys.nPer = 2;               % [] number of sine periods in tx burst
sys.sos = 1540;             % [m/s] speed of sound
sys.att = 0.5e-4;           % [dB/m/Hz]

txAngle = -20:1:20;     % [deg] tx angles (for PWI)
txFocus = 20e-3;        % [m] tx focus (for LIN)
txApert = 32;           % [elem] tx aperture (for LIN)

%% Medium setup: 
if 0
    % tissue-like structure (hours of calculations)
    % parameters
    scatDens	= 5*1e9;                                    % [1/m^3] spatial density of scatterers distribution
    
    zRng        = [ 00 60]*1e-3;                            % [m] range of z-coordinates for the tissue block
    xRng        = [-20 20]*1e-3;                            % [m] range of x-coordinates for the tissue block
    yRng        = [-05 05]*1e-3;                            % [m] range of y-coordinates for the tissue block
    
    % scatterers
    tissueVol	= diff(zRng)*diff(xRng)*diff(yRng);         % [m^3] tissue volume
    scatNum     = round(tissueVol*scatDens);                % [] number of scatterers
    
    rng(7);                                                 % set random number generator to get reproducible results
    
    scat.z	= rand(scatNum,1)*diff(zRng) + zRng(1);     % [m] vector of z-coordinates of scatterers
    scat.x	= rand(scatNum,1)*diff(xRng) + xRng(1);     % [m] vector of x-coordinates of scatterers
    scat.y	= rand(scatNum,1)*diff(yRng) + yRng(1);     % [m] vector of y-coordinates of scatterers
    scat.s	= ones(scatNum,1);                          % [] vector of scattering coefficients
else
    % sparse rectangular grid of point scatterers
    xGrid       = (-15:5:15)*1e-3;      % [m] x-grid vector of point sources
    yGrid       = 0*1e-3;               % [m] y-grid vector of point sources
    zGrid       = (10:5:40)*1e-3;       % [m] z-grid vector of point sources
    
    [scat.x,scat.z,scat.y] = meshgrid(xGrid,zGrid,yGrid);
    scat.s      = ones(size(scat.x));
end
%% Raw rf data simulation

% add path to Field II (this one works for me)
addpath([userpath '\Lib_Field']);

% simulate rf data for SSTA scheme
rfSta = simulateRfSta(sys,scat);

% convert the SSTA data to LIN or PWI format
rfPwi = convertRfSta(rfSta,sys,'pwi',[],[],txAngle,0);
rfLin = convertRfSta(rfSta,sys,'lin',txFocus,txApert,0,0);

%% Raw rf data filtration
filtOrd = 2;
fLo = 0.5*sys.fn;
fHi = 1.5*sys.fn;

[filtB,filtA] = butter(filtOrd,[fLo fHi]/(sys.fs/2),'bandpass');
filtDel = phasez(filtB,filtA,[0 (fLo+fHi)/2],sys.fs)/(2*pi)/sys.fn*sys.fs;
filtDel = -filtDel(2); % [s] filtration time delay for frequency = (fLo+fHi)/2

rfSta = filter(filtB,filtA,rfSta);
rfPwi = filter(filtB,filtA,rfPwi);
rfLin = filter(filtB,filtA,rfLin);

% CPU - 'filter'+IIR is fast enough; 'filtfilt'+IIR or 'filter/convn'+FIR is too slow
% GPU - 'filter'+IIR is fast enough; 'filter'+FIR doesn't work; 'convn'+FIR is quite fast; 'filtfilt' doesn't work;

%% Digital Down Conversion
if 1
    sampPerPrd = 4;
    bandWidth = 1;
    
    cicOrd = 2;
    
    dec = floor((sys.fs/sampPerPrd) / (sys.fn*bandWidth/2));
    
    rfSta = downConv(rfSta,sys,dec,cicOrd);
    rfPwi = downConv(rfPwi,sys,dec,cicOrd);
    rfLin = downConv(rfLin,sys,dec,cicOrd);
    
    sys.fs = sys.fs/dec;
    
    % warning: both filtration and decimation introduce phase shift!
end

%% Image reconstruction
xGrid = (-20:0.1:20)*1e-3;	% [m] 
zGrid = (0:0.05:50)*1e-3;   % [m]

xSize = length(xGrid);
zSize = length(zGrid);

imgRfSta = reconstructRfImg(rfSta,sys,xGrid,zGrid,0,'sta',1,0,0);
imgRfPwi = reconstructRfImg(rfPwi,sys,xGrid,zGrid,0,'pwi',[],[],txAngle);
imgRfLin = reconstructRfImg(rfLin,sys,xGrid,zGrid,0,'lin',txApert,txFocus,0);

rxApert	= 32;
% tx delay of aperture center - far from ideal (does not account for txAngle and tx aperture clipping)
txCentDel= (sqrt(txFocus^2 + (txApert/2*sys.pitch).^2) - txFocus)/sys.sos;
imgRfLin2= reconstructRfLin(rfLin,sys,txCentDel,0,rxApert);

%% Envelope detection
nanMaskSta = isnan(imgRfSta);
nanMaskPwi = isnan(imgRfPwi);
nanMaskLin = isnan(imgRfLin);

imgRfSta(nanMaskSta) = 0;
imgRfPwi(nanMaskPwi) = 0;
imgRfLin(nanMaskLin) = 0;

if isreal(imgRfSta)
    imgRfSta = hilbert(imgRfSta);
end
if isreal(imgRfPwi)
    imgRfPwi = hilbert(imgRfPwi);
end
if isreal(imgRfLin)
    imgRfLin = hilbert(imgRfLin);
end

imgRfSta(nanMaskSta) = nan;
imgRfPwi(nanMaskPwi) = nan;
imgRfLin(nanMaskLin) = nan;

imgEnvSta = abs(imgRfSta);
imgEnvPwi = abs(imgRfPwi);
imgEnvLin = abs(imgRfLin);

%% Compression
imgSta = 20*log10(imgEnvSta);
imgPwi = 20*log10(imgEnvPwi);
imgLin = 20*log10(imgEnvLin);

%% Display
dynRng	= 40;

cLimSta = max(max(imgSta(round(zSize*0.25):round(zSize*0.75),round(xSize*0.25):round(xSize*0.75))));
cLimPwi = max(max(imgPwi(round(zSize*0.25):round(zSize*0.75),round(xSize*0.25):round(xSize*0.75))));
cLimLin = max(max(imgLin(round(zSize*0.25):round(zSize*0.75),round(xSize*0.25):round(xSize*0.75))));

figure;
for n=1:3
    subplot(1,3,n);
    switch n
        case 1
            imagesc(xGrid*1e3,zGrid*1e3,imgSta);
            set(gca,'CLim',cLimSta + [-dynRng 0]);
        case 2
            imagesc(xGrid*1e3,zGrid*1e3,imgPwi);
            set(gca,'CLim',cLimPwi + [-dynRng 0]);
        case 3
            imagesc(xGrid*1e3,zGrid*1e3,imgLin);
            set(gca,'CLim',cLimLin + [-dynRng 0]);
    end
    
    daspect([1 1 1]);
    colormap(gray);
    colorbar;
    xlabel('x [mm]');
    ylabel('z [mm]');
    
end


