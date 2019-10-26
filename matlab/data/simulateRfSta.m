% Script for simulation of SSTA rf-channel data (uses Field II).
% Requires Field II directory to be added to the MATLAB path.

%% Parameters
c           = 1540;                                     % [m/s] speed of sound in the medium
fs          = 40e6;                                     % [Hz] sampling frequency
fn          = 5e6;                                      % [Hz] carrier (nominal) frequency
nPer        = 2;                                        % [] number of periods in the emitted pulse
att0        = 0.0e2;                                    % [dB/m] attenuation at nominal frequency fn
att1        = 0.0e-4;                                   % [dB/m/Hz] attenuation change with frequency

nElem       = 128;                                      % [] number of probe elements
kerf        = 0.0250e-3;                                % [m] kerf
eleXSize	= 0.2798e-3;                                % [m] length of a single probe element in x-direction (width)
eleYSize	= 4.0000e-3;                                % [m] length of a single probe element in y-direction (elevation)
eleXSubNum	= 01;                                       % [] number of sub-elements that a single probe element is divided into in x-direction (width)
eleYSubNum	= 08;                                       % [] number of sub-elements that a single probe element is divided into in y-direction (elevation)

focDepth	= 40e-3;                                    % [m] initial focal depth
focPoint	= [0 0 focDepth];                           % [m] initial focal point

impResp     = sin(2*pi*fn*(0:1/fs:2/fn));               % [] impulse response of probe elements
impResp     = impResp.*hanning(length(impResp))';       % [] impulse response of probe elements
excitation	= sign(sin(2*pi*fn*(0:1/fs:nPer/fn)));      % [] excitation

%% Scatterers setup: tissue-like structure (hours of calculations)
% % Parameters
% scatDens	= 5*1e9;                                    % [1/m^3] spatial density of scatterers distribution
% 
% zRng        = [ 00 60]*1e-3;                            % [m] range of z-coordinates for the tissue block
% xRng        = [-20 20]*1e-3;                            % [m] range of x-coordinates for the tissue block
% yRng        = [-05 05]*1e-3;                            % [m] range of y-coordinates for the tissue block
% 
% % Scatterers
% tissueVol	= diff(zRng)*diff(xRng)*diff(yRng);         % [m^3] tissue volume
% scatNum     = round(tissueVol*scatDens);                % [] number of scatterers
% 
% rng(7);                                                 % set random number generator to get reproducible results
% 
% tissue.z	= rand(scatNum,1)*diff(zRng) + zRng(1);     % [m] vector of z-coordinates of scatterers
% tissue.x	= rand(scatNum,1)*diff(xRng) + xRng(1);     % [m] vector of x-coordinates of scatterers
% tissue.y	= rand(scatNum,1)*diff(yRng) + yRng(1);     % [m] vector of y-coordinates of scatterers
% tissue.s	= ones(scatNum,1);                          % [] vector of scattering coefficients

%% Scatterers setup: sparse rectangular grid of point scatterers
xGrid       = (-15:5:15)*1e-3;      % [m] x-grid vector of point sources
yGrid       = 0*1e-3;               % [m] y-grid vector of point sources
zGrid       = (10:5:40)*1e-3;       % [m] z-grid vector of point sources

[x,z,y]     = meshgrid(xGrid,zGrid,yGrid);
s           = ones(size(x));

%% Field II
% Field initialization
field_init(0);                                          % Initialization

% Set execution parameters
% set_field('threads',            4);                     % Set number of threads (for parallel Field version)
set_field('show_times',         5);                     % Enable remaining time display (flag>2 sets the time between successive reports)
set_field('debug',              0);                     % Enable debug information display
set_field('use_att',            1);                     % Enables the attenuation
set_field('use_rectangles',     1);                     % Use rectangles in the aperture modeling
set_field('use_triangles',      0);                     % Use triangles in the aperture modeling
set_field('use_lines',          0);                     % Use lines in the aperture modeling
set_field('fast_integration',   0);                     % Enables fast integration for bound lines and triangles

% Set physical parameters
set_field('c',                  c);
set_field('att',                att0);
set_field('freq_att',           att1);
set_field('att_f0',             fn);
set_field('fs',                 fs);

% Set transmit and receive apertures
txApert     = xdc_linear_array(nElem,eleXSize,eleYSize,kerf,eleXSubNum,eleYSubNum,focPoint);
rxApert     = xdc_linear_array(nElem,eleXSize,eleYSize,kerf,eleXSubNum,eleYSubNum,focPoint);

% Set impulse responses
xdc_impulse(txApert,impResp);
xdc_impulse(rxApert,impResp);

% Set excitation for the transmit aperture
xdc_excitation(txApert,excitation);

% SSTA simulation, no focusing, no apodization
[rfSta,t0]	= calc_scat_all(txApert,rxApert,[x(:) y(:) z(:)],s(:),1);

% close field
field_end;

%% RF data rearrangement
rfSta       = reshape(rfSta,[],nElem,nElem);

% make the first rx sample be the time t = 0
if t0 <= 0
    rfSta(1:-t0*fs,:,:) = [];
else
    rfSta = [zeros(t0*fs,nElem,nElem); rfSta];
end





