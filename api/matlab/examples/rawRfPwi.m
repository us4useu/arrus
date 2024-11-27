% Example script for raw rf imaging when using plane waves and angular scanning

%% Initialize the system
addpath('../');
addpath('../arrus');

% Make sure the configuration in the *.prototxt file is correct.
us  = Us4R.create('configFile', 'us4r.prototxt');

%% Selected parameters
% To program plane wave sequence, set 'txFocus' to infinity.
txFoc = inf;                    % Focal distance [m]

% To program angular scanning/compounding, set 'txAngle' as a vector. 
% It will also determine the number of transmissions in the sequence.
txAng = (-15:15:15)*pi/180;      % Plane wave angles [rad]
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
                        'txVoltage',        5, ...
                        'rxNSamples',       4*1024, ...
                        'tgcStart',         14, ...
                        'tgcSlope',         200, ...
                        'hwDdcEnable',      false ... % set to false/true to obtain raw data as RF/decimated IQ
                        );

us.upload(seq);

%% Preview (continuous in-loop operation)
% Warning: the raw rf display methods (imageRawRf and plotRawRf) require
% the rf to be real, and thus the hwDdcEnable must be set to false.

% Show image of all rf echoes in the sequence
us.imageRawRf('amplitudeLim',1e3);

% Plot a selected rf line (96 = 32nd line from 2nd TX)
us.plotRawRf('selectedLines',96);

%% Collecting data (single execution of the sequence)
[raw,img] = us.run;



