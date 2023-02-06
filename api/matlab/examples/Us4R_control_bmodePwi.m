% Example script for b-mode imaging using plane waves and angular scanning

%% Initialize the system
addpath('..\');
addpath('..\arrus');

% Make sure the configuration in the *.prototxt file is correct.
us  = Us4R('configFile', 'us4r.prototxt');

%% Selected parameters
% To program plane wave sequence, set 'txFocus' to infinity.
txFoc = inf;                    % Focal distance [m]

% To program angular scanning/compounding, set 'txAngle' as a vector. 
% It will also determine the number of transmissions in the sequence.
txAng = (-15:3:15)*pi/180;      % Plane wave angles [rad]
nTx = numel(txAng);             % Number of transmissions in a sequence

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
                        'txNPeriods',       2, ...
                        'txVoltage',        20, ...
                        'rxDepthRange',     50e-3, ...
                        ... % Optional parameters
                        'hwDdcEnable',      true, ... % set to false/true to obtain raw data as RF/decimated IQ (set false to bypass DDC)
                        'decimation',       10, ... % recommended (and default) value is 65e6/txFrequency
                        'nRepetitions',     1, ...
                        'txPri',            400e-6, ... % time interval between physical transmissions
                        'tgcStart',         14, ...
                        'tgcSlope',         0.02, ...
                        'txInvert',         false ...
                        );

% GPU/CPU reconstruction implemented in matlab.
rec = Reconstruction(   ... % Obligatory parameters
                        'xGrid',            (-20:0.10:20)*1e-3, ...
                        'zGrid',            (  0:0.10:50)*1e-3, ...
                        ... % Optional parameters
                        'gridModeEnable',   true, ... % set to false to reconstruct image lines instead of the full grid for each transmission.
                        'bmodeRxTangLim',   [-0.5 0.5], ...
                        'rxApod',           hamming(10).', ...
                        'bmodeFrames',      1:nTx ...
                        );

us.upload(seq, rec);

%% Preview (continuous in-loop operation)
display = BModeDisplay(rec, 'dynamicRange', [0 80]);
us.runLoop(@display.isOpen, @display.updateImg);

%% Collecting data (single execution of the sequence)
[raw,img] = us.run;

%% Close session
us.closeSession;


