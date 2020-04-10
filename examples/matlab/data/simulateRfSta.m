% Simulation of SSTA rf-channel data (uses Field II).
function[rfSta] = simulateRfSta(sys,acq,pha)
% Outputs:
% 
% rfSta                     - (nSamp,nElem,nElem) output raw rf data for SSTA scheme
% 
% 
% Inputs:
% 
% sys                       - system-related parameters
% sys.nElements             - [elem] number of probe's elements
% sys.pitch                 - [m] probe's pitch
% 
% acq                       - acquisition-related parameters
% acq.rx.samplingFrequency	- [Hz] sampling frequency
% acq.tx.frequency          - [Hz] carrier (nominal) frequency
% acq.tx.nPeriods           - [] number of periods in the emitted pulse
% 
% pha                       - phantom-related parameters
% pha.speedOfSound          - [m/s] speed of sound in the medium
% pha.attenuation           - [dB/m/Hz] linear attenuation coefficient in the medium
% pha.particles.xPosition	- [m] x-coordinates of scattering points
% pha.particles.yPosition	- [m] y-coordinates of scattering points
% pha.particles.zPosition	- [m] z-coordinates of scattering points
% pha.particles.scattering	- [] scattering cross-section of scattering points
% 
% 
% Restrictions:
% 
% requires Field II directory to be added to the MATLAB path.
% pha.particles.xPosition,yPosition,zPosition,scattering must be the same size

%% parameters
kerf        = 0.0000e-3;	% [m] kerf
eleXSize	= sys.pitch;	% [m] length of a single probe element in x-direction (width)
eleYSize	= 4.0000e-3;	% [m] length of a single probe element in y-direction (elevation)
eleXSubNum	= 01;           % [] number of sub-elements that a single probe element is divided into in x-direction
eleYSubNum	= 08;           % [] number of sub-elements that a single probe element is divided into in y-direction

focDepth	= 40e-3;            % [m] initial focal depth
focPoint	= [0 0 focDepth];	% [m] initial focal point

impResp     = sin(2*pi*acq.tx.frequency*(0:1/acq.rx.samplingFrequency:2/acq.tx.frequency));
impResp     = impResp.*hanning(length(impResp))';       % [] impulse response of probe elements
excitation	= sign(sin(2*pi*acq.tx.frequency*(0:1/acq.rx.samplingFrequency:acq.tx.nPeriods/acq.tx.frequency)));	% [] excitation

%% Field II
% Field initialization
field_init(0);                      % Initialization

% set execution parameters
% set_field('threads',            4);	% Set number of threads (for parallel Field version)
set_field('show_times',         5);	% Enable remaining time display (flag>2 sets the time between successive reports)
set_field('debug',              0);	% Enable debug information display
set_field('use_att',            1);	% Enables the attenuation
set_field('use_rectangles',     1);	% Use rectangles in the aperture modeling
set_field('use_triangles',      0);	% Use triangles in the aperture modeling
set_field('use_lines',          0);	% Use lines in the aperture modeling
set_field('fast_integration',   0);	% Enables fast integration for bound lines and triangles

% set physical parameters
set_field('c',                  pha.speedOfSound);
set_field('att',                pha.attenuation*acq.tx.frequency);	% [dB/m] attenuation at tx frequency
set_field('freq_att',           pha.attenuation);                   % [dB/m/Hz] attenuation change with frequency
set_field('att_f0',             acq.tx.frequency);
set_field('fs',                 acq.rx.samplingFrequency);

% set transmit and receive apertures
txApert     = xdc_linear_array(sys.nElements,eleXSize,eleYSize,kerf,eleXSubNum,eleYSubNum,focPoint);
rxApert     = xdc_linear_array(sys.nElements,eleXSize,eleYSize,kerf,eleXSubNum,eleYSubNum,focPoint);

% set impulse responses
xdc_impulse(txApert,impResp);
xdc_impulse(rxApert,impResp);

% set excitation for the transmit aperture
xdc_excitation(txApert,excitation);

% SSTA simulation, no focusing, no apodization
[rfSta,t0]	= calc_scat_all(txApert,rxApert,[pha.particles.xPosition(:) ...
                                             pha.particles.yPosition(:) ...
                                             pha.particles.zPosition(:)], ...
                                             pha.particles.scattering(:),1);

% close field
field_end;

%% rf data rearrangement
rfSta       = reshape(rfSta,[],sys.nElements,sys.nElements);

% make the first rx sample be the time t = 0
if t0 <= 0
    rfSta(1:-t0*acq.rx.samplingFrequency,:,:) = [];
else
    rfSta = [zeros(t0*acq.rx.samplingFrequency,sys.nElements,sys.nElements); rfSta];
end

end



