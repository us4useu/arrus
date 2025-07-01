% Example script for switching Tx voltage dynamically between Tx events.
% 
% Tx voltage does not have to be set constant for the whole sequence. 
% It can also be selected from between two predefined different levels, 
% for each Tx event individually. This example shows how to do this.

% The sequence used in this example has two different Tx voltage levels:
% <-10, +10> and <-20, +20> [V]. The first one is used for Tx events #1-3 
% (txAngle = -10, 0, and 10 deg) and the second is used for Tx events #4-7
% (txAngle = 0, 0, 0, and 0 deg). 
% 
% NOTE: Tx voltage levels must meet the following rule: 
%   all(txVoltage(1,:) < txVoltage(2,:))

%% Initialize the system
addpath('../');
addpath('../arrus');

% Make sure the configuration in the *.prototxt file is correct.
us  = Us4R.create('configFile', 'us4r.prototxt');

%% Tx/Rx sequences and reconstruction parameters
seq = CustomTxRxSequence( 'txApertureCenter', 0, ...
                          'txApertureSize',   us.getNProbeElem, ...
                          'rxApertureCenter', 0, ...
                          'rxApertureSize',   us.getNProbeElem, ...
                          'txFocus',          inf, ...
                          'txAngle',          [-10 0 10 0 0 0 0]*pi/180, ...
                          'speedOfSound',     1540, ...
                          'txFrequency',      6.5e6, ...
                          'txNPeriods',       2, ...
                          'txVoltage',        [10, 10; ...             % predefined Tx voltage level #1 (negative and positive voltage)
                                               20, 20], ...            % predefined Tx voltage level #2 (negative and positive voltage)
                          'txVoltageId',      [1 1 1 2 2 2 2], ...     % select Tx voltage level for each Tx individually
                          'rxDepthRange',     50e-3, ...
                          'hwDdcEnable',      false, ...
                          'tgcStart',         14, ...
                          'tgcSlope',         200 ...
                          );

us.upload(seq);

%% Live preview of all rf echoes in the sequence
us.imageRawRf('amplitudeLim',10e3);

