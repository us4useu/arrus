% Example script for b-mode & color doppler imaging (duplex) using plane waves

%% Initialize the system
addpath('../');
addpath('../arrus');

% Make sure the configuration in the *.prototxt file is correct.
us  = Us4R('configFile', 'us4r.prototxt');

%% Selected parameters
% To program plane wave sequence, set 'txFocus' to infinity.
txFoc = inf;                    % Focal distance [m]

% Set different tx angles for B-Mode and Color Doppler
txAngBMode = (-15:3:15)*pi/180; % Plane wave angles for B-Mode [rad]
nTxBMode = numel(txAngBMode);   % Number of transmissions for B-Mode
    
txAngColor = 10*pi/180;         % Plane wave angle for Color [rad]
nTxColor = 64;                  % Number of transmissions for Color

txAng = [txAngBMode, txAngColor*ones(1,nTxColor)];

% Set different pulse lengths for B-Mode and Color Doppler
txNPerBMode = 2;
txNPerColor = 8;
txNPer = [txNPerBMode*ones(1,nTxBMode), ...
          txNPerColor*ones(1,nTxColor)];

% Set rx tangent limits for Color Doppler (account for angled rx beam)
rxTangLimColor = [-0.5 0.5] + tan(txAngColor);

% Wall Clutter Filter parameters
[wcfB, wcfA]    = butter(8,0.30,'high'); % Wall clutter filter coefficients
wcfInitSize     = 32;

%% Program Tx/Rx sequence and reconstruction
% Set the Tx/Rx sequence
% If the sequence consists of nTx transmissions, then the following
% parameters must be vectors of length nTx or scalars (if they are constant):
% txApertureCenter/txCenterElement, rxApertureCenter/rxCenterElement,
% txApertureSize, txFocus, txAngle, txFrequency, txNPeriods, txInvert.
% All other parameters are constant for the whole sequence, thus they are scalars.
seq = CustomTxRxSequence(... % Obligatory parameters
                        ... % (tx/rxApertureCenter [m] can be replaced with 
                        ... % tx/rxCenterElement [elem], rxDepthRange [m] 
                        ... % can be replaced with rxNSamples [samp])
                        'txApertureCenter', 0e-3, ...
                        'txApertureSize',   us.getNProbeElem, ...
                        'rxApertureCenter', 0e-3, ...
                        'rxApertureSize',   us.getNProbeElem, ...
                        'txFocus',          txFoc, ...
                        'txAngle',          txAng, ...
                        'speedOfSound',     1540, ...
                        'txFrequency',      6.5e6, ...
                        'txNPeriods',       txNPer, ...
                        'txVoltage',        10, ...
                        'rxDepthRange',     50e-3, ...
                        ... % Optional parameters
                        'hwDdcEnable',      true, ... % set to false/true to obtain raw data as RF/decimated IQ (set false to bypass DDC)
                        'decimation',       10, ... % recommended (and default) value is round(65e6/txFrequency)
                        'nRepetitions',     1, ...
                        'txPri',            120e-6, ... % tx time interval kept low for Doppler
                        'tgcStart',         14, ...
                        'tgcSlope',         200, ...
                        'txInvert',         false ...
                        );

% GPU/CPU reconstruction implemented in matlab.
rec = Reconstruction(   ... % Obligatory parameters
                        'xGrid',            (-20:0.10:20)*1e-3, ...
                        'zGrid',            (  0:0.10:30)*1e-3, ...
                        ... % Optional parameters
                        'gridModeEnable',   true, ... % set to false to reconstruct image lines instead of the full grid for each transmission.
                        'bmodeRxTangLim',   [-0.5 0.5], ...
                        'rxApod',           hamming(10).', ...
                        'bmodeFrames',      1:nTxBMode, ...
                        ... % Color Doppler parameters
                        'colorEnable',      true, ...
                        'colorFrames',      (1:nTxColor) + nTxBMode, ...
                        'colorRxTangLim',	rxTangLimColor, ...
                        'wcFilterBCoeff',   wcfB, ...
                        'wcFilterACoeff',	wcfA, ...
                        'wcFiltInitSize',	wcfInitSize);

us.upload(seq, rec);

%% Preview (continuous in-loop operation)
display = DuplexDisplay(rec, 'dynamicRange',    [0 80], ...
                             'powerThreshold',  10, ...   % Color Doppler is shown if Power is ABOVE this level [dB]
                             'turbuThreshold',  0.99, ... % Color Doppler is shown if Turbulence is BELOW this level (value range: 0-1)
                             'stdevThreshold',  2.60, ... % Color Doppler is shown if Color St. Dev. is BELOW this level (value range: 0-pi)
                             'thresholdSmooth', 20);
us.runLoop(@display.isOpen, @display.updateImg);

%% Collecting data (single execution of the sequence)
[raw,img] = us.run;

%% Close session
us.closeSession;


