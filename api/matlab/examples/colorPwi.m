% Example script for b-mode & color doppler imaging (duplex) using plane waves

% There are several approaches to design the Tx/Rx sequence for duplex
% mode. They all must ensure that the Color Doppler frames are acquired 
% at a constant pace (preferably as high as possible). Some basic examples
% of duplex sequences using plane waves are listed below:
% 
% 1. Transmit N times at a constant Tx angle. 
% B-mode is reconstructed from any of the collected frames. 
% Color Doppler is reconstructed from all of the collected frames. 
% This approach provides the highest Color Doppler PRF.
% 
% 2. Transmit N times (N is even).
% B-mode is reconstructed from the odd frames which use various Tx angles.
% Color Doppler is reconstructed from the even frames which use a constant
% Tx angle.
% This approach provides much better B-mode image (compouning of frames
% acquired for a range of Tx angles) for the price of reduced Color Doppler
% quality (reduced PRF, less frames for Color Doppler estimation).
% 
% The below code makes possible to test both approaches by setting the
% exampleId to 1 or 2, respectively.
% 
% Tips:
% - To improve Color Doppler SNR, use longer pulses (txNPeriods 4-8). In
% example #1 this negatively affects B-mode resolution but in example #2
% B-mode & Color Doppler frames are separable.
% - Color Doppler sensitivity decreases if the effective beam direction
% gets close to perpendicular to the flow direction. Most imaged vessels
% are parallel to the skin/probe surface i.e. perpendicular to the beam if 
% it is not angled. It is therefore advisable to angle the Tx beam for the
% Color Doppler frames. The same applies to the Rx angle which is steered
% by the colorRxTangLim property of Reconstruction.
% - Color Doppler works best for uninterrupted multiple execution of the
% Tx/Rx sequence. For this purpose set workMode to 'SYNC' and make sure 
% that your hardware is capable to transfer & process a data frame in 
% real time i.e. the time required for acquisition of another data frame.
% If not, this will affect not only the Duplex display frame rate (time 
% lags) but also the Color Doppler quality (included filter needs
% initialisation if time regime is broken). To handle this, txPri should
% be gradually increased until the lags disappear.

%% Initialize the system
addpath('../');
addpath('../arrus');

% Make sure the configuration in the *.prototxt file is correct.
us  = Us4R.create('configFile', 'us4r.prototxt');

%% Selected parameters
exampleId = 1;  % set to 1 or 2 for running example #1 or #2, respectively

nTx = 32;       % number of Tx/Rx frames in the sequence

switch exampleId
    case 1
        iTxBmode = 1;           % first frame for B-mode
        iTxColor = 1:nTx;       % all frames for Color Doppler
        
        txAngColor = 20 * pi/180;           % constant Tx angle, try to avoid beam-flow perpendicularity -> 15deg
        txAng = txAngColor * ones(1,nTx);   % at least one sequence parameter must have nTx elements
        
        txNPer = 4;             % constant Tx pulse length, a compromise between B-mode resolution and Color Doppler SNR
        
    case 2
        iTxBmode = 1:2:nTx;     % odd frames for B-mode
        iTxColor = 2:2:nTx;     % even frames for Color Doppler
        
        % Set Tx angles
        txAngBmode = linspace(-20,20,numel(iTxBmode)) * pi/180; % a range of Tx angles for angular compounding
        txAngColor = 20 * pi/180;   % constant Tx angle, try to avoid beam-flow perpendicularity -> 15deg
        
        txAng = zeros(1,nTx);
        txAng(iTxBmode) = txAngBmode;
        txAng(iTxColor) = txAngColor;
        
        % Set Tx pulse lengths
        txNPerBMode = 2;        % short pulse for better resolution
        txNPerColor = 8;        % long pulse for better SNR
        
        txNPer = zeros(1,nTx);
        txNPer(iTxBmode) = txNPerBMode;
        txNPer(iTxColor) = txNPerColor;

    otherwise
        error('Invalid exampleId');
end

% Set Rx tangent limits
rxTangLimBmode = [-1.0 1.0];                        % wide Rx aperture for better resolution
rxTangLimColor = [-0.5 0.5] + tan(txAngColor);      % narrow Rx aperture for better distinction of beam direction
                                                    % moved Rx aperture, try to avoid beam-flow perpendicularity

% Wall Clutter Filter parameters
wcfOrder = 8;       % filter order (max 8 order supported)
wcfCutoff = 0.30;   % filter cutoff (fc/(fs/2))
wcfInitSize = 8;    % number of filter output samples skipped after filter sturtup

[wcfB, wcfA] = butter(wcfOrder, wcfCutoff, 'high');

%% Program Tx/Rx sequence and reconstruction
% Set the Tx/Rx sequence
% If the sequence consists of nTx transmissions, then the following
% parameters must be vectors of length nTx or scalars (if they are constant):
% txApertureCenter/txCenterElement, rxApertureCenter/rxCenterElement,
% txApertureSize, txFocus, txAngle, txFrequency, txNPeriods, txInvert.
% All other parameters are constant for the whole sequence, thus they are scalars.
seq = CustomTxRxSequence(... 
                        'txApertureCenter', 0e-3, ...
                        'txApertureSize',   us.getNProbeElem, ...
                        'rxApertureCenter', 0e-3, ...
                        'rxApertureSize',   us.getNProbeElem, ...
                        'txFocus',          inf, ...    % for plane waves set 'txFocus' to infinity
                        'txAngle',          txAng, ...
                        'speedOfSound',     1540, ...
                        'txFrequency',      4.0e6, ...
                        'txNPeriods',       txNPer, ...
                        'txVoltage',        15, ...
                        'rxDepthRange',     30e-3, ...
                        ... % Optional parameters
                        'txPri',            200e-6, ...
                        'tgcStart',         14, ...
                        'tgcSlope',         200, ...
                        'workMode',         'SYNC', ...
                        'bufferSize',       16 ...
                        );

% GPU/CPU reconstruction implemented in matlab.
rec = Reconstruction(   ... % Obligatory parameters
                        'xGrid',            (-15:0.20:15)*1e-3, ...
                        'zGrid',            (  5:0.20:25)*1e-3, ...
                        ... % Optional parameters
                        'bmodeRxTangLim',   rxTangLimBmode, ...
                        'rxApod',           hamming(10).', ...
                        'bmodeFrames',      iTxBmode, ...
                        ... % Color Doppler parameters
                        'colorEnable',      true, ...
                        'colorFrames',      iTxColor, ...
                        'colorRxTangLim',	rxTangLimColor, ...
                        'wcFilterBCoeff',   wcfB, ...
                        'wcFilterACoeff',	wcfA, ...
                        'wcFiltInitSize',	wcfInitSize);

us.upload(seq, rec);

%% Preview (continuous in-loop operation)
display = DuplexDisplay(rec, 'dynamicRange',    [0 80], ...
                             'powerThreshold',  20, ...   % Color Doppler is shown if Power is ABOVE this level [dB]
                             'turbuThreshold',  0.50, ... % Color Doppler is shown if Turbulence is BELOW this level (value range: 0-1)
                             'stdevThreshold',  3.50, ... % Color Doppler is shown if Color St. Dev. is BELOW this level (value range: 0-pi)
                             'thresholdSmooth', 3);

us.runLoop(@display.isOpen, @display.updateImg);

%% Collecting data (single execution of the sequence)
[raw,img] = us.run;



