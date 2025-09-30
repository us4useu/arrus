% Example script for quick switching between sequences.
% 
% Sequence upload is currently a time-consuming process. If you need two 
% different sequences to be run one after the other, and you want the
% switching process to be quick, there is a feature that makes it possible.
% It allows you to upload all your sequences at once. Then you just need to
% select which subsequence you want to use. This example shows how to do it.

%% Initialize the system
addpath('../');
addpath('../arrus');

% Make sure the configuration in the *.prototxt file is correct.
us  = Us4R.create('configFile', 'us4r.prototxt');

%% Tx/Rx sequences and reconstruction parameters
% Two sequences are defined:
% seq1 - 11 plane waves, Tx angles = -15:3:15 [deg]
% seq2 - 100 plane waves, Tx angle = 0 [deg]
% Let us assume that we want the seq1 to be used for preview (compounding 
% allows us to better identify the region of interest), while seq2 is 
% actually the sequence that we want to use for data collection.
% 
% We will also use a simple reconstruction operation defined by object rec1.

seq1 = CustomTxRxSequence('txApertureCenter', 0, ...
                          'txApertureSize',   us.getNProbeElem, ...
                          'rxApertureCenter', 0, ...
                          'rxApertureSize',   us.getNProbeElem, ...
                          'txFocus',          inf, ...
                          'txAngle',          (-15:3:15)*pi/180, ... % 11 Tx angles
                          'speedOfSound',     1540, ...
                          'txFrequency',      6.5e6, ...
                          'txNPeriods',       2, ...
                          'txVoltage',        10, ...
                          'rxDepthRange',     50e-3, ...
                          'tgcStart',         14, ...
                          'tgcSlope',         200, ...
                          'hwDdcEnable',      false ...
                          );

seq2 = CustomTxRxSequence('txApertureCenter', 0, ...
                          'txApertureSize',   us.getNProbeElem, ...
                          'rxApertureCenter', 0, ...
                          'rxApertureSize',   us.getNProbeElem, ...
                          'txFocus',          inf, ...
                          'txAngle',          zeros(1,100), ... % 1 Tx angle, 100 times
                          'speedOfSound',     1540, ...
                          'txFrequency',      6.5e6, ...
                          'txNPeriods',       2, ...
                          'txVoltage',        10, ...
                          'rxDepthRange',     50e-3, ...
                          'tgcStart',         14, ...
                          'tgcSlope',         200, ...
                          'hwDdcEnable',      false ...
                          );

rec1 = Reconstruction('xGrid',            (-20:0.10:20)*1e-3, ...
                      'zGrid',            (  0:0.10:40)*1e-3, ...
                      'bmodeRxTangLim',   [-1.0 1.0], ...
                      'rxApod',           hamming(21).' ...
                      );

%% Preview and data acquisition
if false
    %% The old (time-consuming) way of switching from seq1 to seq2
    
    % Upload 1st sequence
    us.upload(seq1, rec1);
    
    % Preview
    display = BModeDisplay(rec1, 'dynamicRange', [0 80]);
    us.runLoop(@display.isOpen, @display.updateImg);
    
    % Upload 2nd sequence
    tic;
    us.upload(seq2);
    disp(['Time lag due to uploading new sequence is ' num2str(toc) 's']);
    
    % Data acquisition
    raw = us.run;
    
else
    %% The new (quick) way of switching from seq1 to seq2
    
    % Upload both sequences at once. Unlike upload method, uploadSequence can do that.
    us.uploadSequence([seq1, seq2]);
    
    % Select sequence seq1 (it was the 1st one in the vector of uploaded sequences).
    us.selectSequence(1);
    
    % Upload the reconstruction parameters. It will have to be re-uploaded
    % after each selectSequence call, unless no reconstruction is needed 
    % for the selected sequence.
    us.setReconstruction(rec1);
    
    % Preview
    display = BModeDisplay(rec1, 'dynamicRange', [0 80]);
    us.runLoop(@display.isOpen, @display.updateImg);
    
    % Select sequence seq2 (it was the 2nd one in the vector of uploaded sequences).
    tic;
    us.selectSequence(2);
    disp(['Time lag due to selecting new sequence is ' num2str(toc) 's']);

    % We just need to acquire raw data, no need to call the setReconstruction method.
    
    % Data acquisition
    raw = us.run;
    
    % Back to preview after the data collection:
    us.selectSequence(1);
    us.setReconstruction(rec1);
    display = BModeDisplay(rec1, 'dynamicRange', [0 80]);
    us.runLoop(@display.isOpen, @display.updateImg);

end

