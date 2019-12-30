% Simulation of simple Doppler signal from a circular vessel

%% Parameters
fn = 5e6;                       % [Hz] carrier frequency
sos = 1540;                     % [m/s] speed of sound

prf = 10e3;                     % [Hz] pulse repetition frequency
nRep = 1024;                    % [] number of repetitions

% Grid
xGrid = (-20:0.1:20)*1e-3;      % [m] x-grid vector
zGrid = (0:0.1:40)*1e-3;        % [m] z-grid vector

% Flow information
flowVelMax = 0.25;              % [m/s] max flow velocity
flowWidth = 5e-3;               % [m] width (diameter) of the flow
flowRadius = 10e-3;             % [m] radius of curvature of the flow
flowXCenter = 0e-3;             % [m] x-position of the flow
flowZCenter = 20e-3;            % [m] z-position of the flow

% Amplitudes
aClutter = 60;                  % [dB] amplitude of clutter signal
aDoppler = 20;                  % [dB] amplitude of doppler signal
aNoiseBeg = 0;                  % [dB] amplitude of noise (starting)
aNoiseEnd = 20;                 % [dB] amplitude of noise (ending)

%% Calculations
% [m] radial distance from the center of the flow
rad = sqrt((xGrid - flowXCenter).^2 + (zGrid - flowZCenter)'.^2);

% cosine of the flow-beam angle
angCos = (xGrid - flowXCenter)./max(rad,1e-6);

% ratio of the flow velocity to max flow velocity due to parabolic flow profile
velProf = max(0,(1 - ((rad - flowRadius)/(flowWidth/2)).^2));

% [m/s] flow velocity
vel = flowVelMax*velProf;

% [Hz] Doppler frequency
fd = fn.*(2*vel/sos).*angCos;

% [s] time vectors
tFast = 2*zGrid/sos;
tSlow = (0:(nRep-1))/prf;

% signal components
iqClutter = 10^(aClutter/20)*exp(1i*2*pi*fn*tFast');
iqDoppler = 10^(aDoppler/20)*exp(1i*2*pi*fd.*permute(tSlow,[1 3 2])).*double(vel>0);
iqNoise = logspace(aNoiseBeg/20,aNoiseEnd/20,length(zGrid))'.* ...
            complex(randn(size(iqDoppler)),randn(size(iqDoppler)));

% output signal
iqData = iqClutter + iqDoppler + iqNoise;


