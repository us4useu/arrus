% Example script for long acquisition.
% 
% Some measurements may require collecting data at a high 
% and undisturbed rate over a long period of time. 
% There is a limit to the number of Tx/Rx events in a sequence.
% If the acquisition is short enough to be covered by a single Tx/Rx
% sequence, and proper workMode is used (i.e. 'SYNC' or 'ASYNC') the time 
% regime is maintained by design. Longer acquisitions may require a 
% repeated executions of the sequence. 
% Data acquired for a single sequence execution are stored in a data buffer,
% occupying a single buffer element. Transfering the data from the buffer 
% element to the host PC frees the buffer element. If the data are
% collected faster than they are transferred to the host PC, at some point
% there will be no free buffer element for the new data to be stored into 
% (buffer overflow). This, in turn, will result in breaking the time regime.
% The system can either wait for the buffer to be emptied, or can overwrite 
% the data that were not transferred to the host PC yet (in Matlab this will 
% cause an error). This example shows how to deal with this problem.

%% Initialize the system
addpath('../');
addpath('../arrus');

% Make sure the configuration in the *.prototxt file is correct.
us  = Us4R.create('configFile', 'us4r.prototxt');

%% Tx/Rx sequences and reconstruction parameters

% A typical plane wave sequence. The parameters that are important from
% the point of view of this example (txPri, workMode, bufferSize) will be
% discused at the end of this script.
seq = CustomTxRxSequence( 'txApertureCenter', 0, ...
                          'txApertureSize',   us.getNProbeElem, ...
                          'rxApertureCenter', 0, ...
                          'rxApertureSize',   us.getNProbeElem, ...
                          'txFocus',          inf, ...
                          'txAngle',          (-10:2:10)*pi/180, ... % discussed below (4)
                          'speedOfSound',     1540, ...
                          'txFrequency',      6.5e6, ...
                          'txNPeriods',       2, ...
                          'txVoltage',        10, ...
                          'rxDepthRange',     50e-3, ...
                          'tgcStart',         14, ...
                          'tgcSlope',         200, ...
                          'workMode',         'SYNC', ... % discussed below (1)
                          'bufferSize',       8, ...      % discussed below (2)
                          'txPri',            500e-6 ...  % discussed below (3)
                          );

rec = Reconstruction( 'xGrid',            (-20:0.10:20)*1e-3, ...
                      'zGrid',            (  0:0.10:40)*1e-3, ...
                      'bmodeRxTangLim',   [-1.0 1.0], ...
                      'rxApod',           hamming(21).' ...
                      );

us.upload(seq, rec);

%% Preview and acquisition

% The runLoop method offers a cineloop buffer functionality. This buffer is
% limited by the amount of available RAM on the host PC. 

display = BModeDisplay(rec, 'dynamicRange', [0 80]);

[raw, img, sri] = us.runLoop(@display.isOpen, ...
                             @display.updateImg, ...
                             'bufferType', 'all', ...  % discussed below (5), can be 'none','raw','img','all'
                             'bufferMode', 'subs', ... % discussed below (6), can be 'conc' or 'subs'
                             'bufferSize', 1000 ...    % discussed below (7)
                             );

figure;
plot(sri); % discussed below (8)

%% DISCUSSION
% 1 - to maintain the time regime between successive executions of a 
% sequence, its 'workMode' property must be set to 'SYNC' or 'ASYNC'. 
% For SYNC mode, if buffer overflow occurs, the system waits until 
% all buffer elements are freed (long time lag). For ASYNC mode, the 
% system overwrites the data (in Matlab it results in an error).
% 
% 2 - the bufferSize value itself does not prevent from the overflow.
% However, as the data transfer to the host PC is not deterministic and 
% can vary significantly, larger buffer size can help absorb the temporary 
% problems with the transfer.
% NOTE: increasing the bufferSize reduces the max. sequence length 
% and increases the upload time.
% 
% 3 - from the point of view of long and undisturbed acquisitions, the 
% data transfer time cannot be longer than acquisition time. At some point 
% some increase in the txPri may be required.
% 
% 4 - in case of very short sequences it may be reasonable to increase 
% their length (e.g. by repeating the txAngles). This makes the data 
% packages larger and transfers less frequent, which may increase the 
% overall transfer rate.
% 
% 5 - the bufferType allows you to select data type to be buffered: can be 
% raw data and/or image data.
% 
% 6 - bufferMode, if set to 'conc' (concurrent), provides a preview 
% all along the acquisition. This, however, requires additional time 
% for reconstruction, and thus may require increase in txPri or may lead 
% to system buffer overflow. In this mode, the buffer operates as circular.
% If the bufferMode is set to 'subs' (subsequent), then the preview 
% is available before the data acquisition to the buffer. During the 
% preview, the risk of system buffer overflow is still increased due 
% to the reconstruction. After closing the preview window, the data 
% are only acquired and transfered. Reconstruction, if needed, is done 
% after the acquisition, which helps to keep the acquisition time regime. 
% In this mode the acquisition lasts until the buffer is full.
% 
% 7 - bufferSize can be increased to extend the time duration of the
% measurement. It is limited by the amount of the available RAM on 
% the host PC.
% 
% 8 - the sri (sequence repeting interval) can be checked for occurrence of
% time lags.

