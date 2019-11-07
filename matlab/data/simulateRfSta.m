% Simulation of SSTA rf-channel data (uses Field II).
% Requires Field II directory to be added to the MATLAB path.
function[rfSta] = simulateRfSta(sys,scat)
% Outputs:
% rfSta - (nSamp,nElem,nElem) output raw rf data for SSTA scheme

% Inputs:
% sys - system-related parameters
% sys.nElem - [elem] number of transducer elements
% sys.pitch - [m] transducer pitch
% sys.fs - [Hz] sampling frequency
% sys.fn - [Hz] carrier (nominal) frequency
% sys.nPer - [] number of periods in the emitted pulse
% sys.sos - [m/s] assumed speed of sound in the medium
% sys.att - [dB/m/Hz] assumed linear attenuation coefficient in the medium
% scat - medium characterisation (scattering points)
% scat.x - [m] x-coordinates of scattering points
% scat.y - [m] y-coordinates of scattering points
% scat.z - [m] z-coordinates of scattering points
% scat.s - [] scattering cross-section of scattering points

% scat.x,y,z,s must be the same size

%% parameters
kerf        = 0.0000e-3;                                % [m] kerf
eleXSize	= sys.pitch;                                % [m] length of a single probe element in x-direction (width)
eleYSize	= 4.0000e-3;                                % [m] length of a single probe element in y-direction (elevation)
eleXSubNum	= 01;                                       % [] number of sub-elements that a single probe element is divided into in x-direction (width)
eleYSubNum	= 08;                                       % [] number of sub-elements that a single probe element is divided into in y-direction (elevation)

focDepth	= 40e-3;                                    % [m] initial focal depth
focPoint	= [0 0 focDepth];                           % [m] initial focal point

impResp     = sin(2*pi*sys.fn*(0:1/sys.fs:2/sys.fn));               % [] impulse response of probe elements
impResp     = impResp.*hanning(length(impResp))';       % [] impulse response of probe elements
excitation	= sign(sin(2*pi*sys.fn*(0:1/sys.fs:sys.nPer/sys.fn)));      % [] excitation

%% Field II
% Field initialization
field_init(0);                                          % Initialization

% set execution parameters
% set_field('threads',            4);                     % Set number of threads (for parallel Field version)
set_field('show_times',         5);                     % Enable remaining time display (flag>2 sets the time between successive reports)
set_field('debug',              0);                     % Enable debug information display
set_field('use_att',            1);                     % Enables the attenuation
set_field('use_rectangles',     1);                     % Use rectangles in the aperture modeling
set_field('use_triangles',      0);                     % Use triangles in the aperture modeling
set_field('use_lines',          0);                     % Use lines in the aperture modeling
set_field('fast_integration',   0);                     % Enables fast integration for bound lines and triangles

% set physical parameters
set_field('c',                  sys.sos);
set_field('att',                sys.att*sys.fn);        % [dB/m] attenuation at nominal frequency fn
set_field('freq_att',           sys.att);               % [dB/m/Hz] attenuation change with frequency
set_field('att_f0',             sys.fn);
set_field('fs',                 sys.fs);

% set transmit and receive apertures
txApert     = xdc_linear_array(sys.nElem,eleXSize,eleYSize,kerf,eleXSubNum,eleYSubNum,focPoint);
rxApert     = xdc_linear_array(sys.nElem,eleXSize,eleYSize,kerf,eleXSubNum,eleYSubNum,focPoint);

% set impulse responses
xdc_impulse(txApert,impResp);
xdc_impulse(rxApert,impResp);

% set excitation for the transmit aperture
xdc_excitation(txApert,excitation);

% SSTA simulation, no focusing, no apodization
[rfSta,t0]	= calc_scat_all(txApert,rxApert,[scat.x(:) scat.y(:) scat.z(:)],scat.s(:),1);

% close field
field_end;

%% rf data rearrangement
rfSta       = reshape(rfSta,[],sys.nElem,sys.nElem);

% make the first rx sample be the time t = 0
if t0 <= 0
    rfSta(1:-t0*sys.fs,:,:) = [];
else
    rfSta = [zeros(t0*sys.fs,sys.nElem,sys.nElem); rfSta];
end

end



