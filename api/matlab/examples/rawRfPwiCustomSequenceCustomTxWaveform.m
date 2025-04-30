%% README
% This script demonstrates how to program us4R-lite in order to run a
% custom TX/RX sequence, and then collect raw RF data. This script runs 
% a sequence of two TX/RX operations and displays both collected RF frames 
% in real-time.

% See the comments below for more detals.


addpath("../");
addpath("../arrus");

import arrus.session.*;
import arrus.ops.us4r.*;


session = [];
try
    %% Initialization and configuration

    % Initialize ARRUS package: specify logging level and the log output
    % file.
    arrus.initialize("clogLevel", "INFO", "logFilePath", "arrus.log", "logFileLevel", "TRACE");
    % Start a new communication session with device. You need to provide
    % here the path to the .prototxt configuration file. 
    session = arrus.session.Session("us4r.prototxt");
    % Get a handle to the us4R/us4R-lite device. 
    us4r = session.getDevice("/Us4R:0");
    % Set the TX voltage. 
    % +/- 10 V
    us4r.setVoltage(10);

    %% TX/RX sequence programming

    % Get the number elements of the probe. 
    probe = us4r.getProbeModel;
    nElements = probe.nElements;
    
    rxApertureSize = nElements;
    nSamples = 2048;
    txAperture = ones(1, nElements);
    rxAperture = ones(1, rxApertureSize);

    waveformBuilder = arrus.ops.us4r.WaveformBuilder();
    wf = waveformBuilder.add([0.2e-6, 0.5e-6, 1e-6], [-1, 1, -1], 2) ...
                        .add([1.5e-6, 2e-6,   3e-6], [1,  0,  1], 1) ...
                        .build();

    nTx = 2;
    % Define the first TX/RX
    % - aperture: an binary mask (an array of 0s and 1s); aperture[i] == 1 means
    %   that the i-th element should be on during TX or RX. 
    % - delays: an array of TX delays [seconds]
    % - sampleRange: the sampling range. The sample range [start, end] 
    %   during RF signal acquisition by the ADC (with the assumed hardware 
    %   sampling frequency, e.g., 65 or 120 MHz). The start time == 0 
    %   corresponds to the moment of transmission initiation 
    %   (tx delay == 0). The end time represents the maximum imaging depth. 
    % - pri: pulse repetition interval (the inverse of pulse repetition 
    %   frequency, PRF. The amount of time that TX/RX 
    %   (transmission/reception) should take [seconds].
    tx = Tx("aperture", txAperture, "delays", zeros(1, nElements), "pulse", wf);
    rx = Rx("aperture", rxAperture, "sampleRange", [0 nSamples]);
    ops(1) = TxRx("tx", tx, "rx", rx, "pri", 500e-6);

    % Define the second TX/RX 
    tx = Tx("aperture", txAperture, "delays", zeros(1, nElements), "pulse", wf);
    rx = Rx("aperture", rxAperture, "sampleRange", [0 nSamples]);
    ops(2) = TxRx("tx", tx, "rx", rx, "pri", 500e-6);

    % Define the TX/RX sequence
    sequence = TxRxSequence("ops", ops);
    % Define the TX/RX and processing scheme: here we are just running
    % a the TX/RX sequence.
    scheme = Scheme("txRxSequence", sequence, "workMode", "MANUAL");
    
    % Upload the sequence on the us4R/us4R-lite.
    uploadResult = cell(1, 7);
    [uploadResult{:}] = session.upload(scheme);

    buffer = uploadResult{1};
    reorderRf = RemapToLogicalOrder(scheme, uploadResult(2:end));
    
    
    %% Display RF data in real-time.
    amplitudeLim = 10000;
    nLines = rxApertureSize*nTx;
    selectedLines = 1:nLines;
    hFig = figure();
    hDisp = imagesc(1:nLines, 1:nSamples, []);
    xlabel('rf line #');
    ylabel('sample #');
    set(gca, 'XLim', [0.5 nLines+0.5]);
    set(gca, 'YLim', [0.5 nSamples+0.5]);
    set(gca,'CLim', amplitudeLim*[-1 1]);
    colormap(jet);
    colorbar;   
    while(ishghandle(hFig))
        % Run the TX/RX sequence on the device.
        session.run();  
        % Get the RF data.
        % The collected data is in the physical order determined by the 
        % system's structure.
        rf = buffer.front().eval();
        rf = rf(:, :);
        % Process the data into a logical order, i.e., determined with 
        % respect to the elements of the probe. 
        % The table below has the following dimensions:
        % (number of samples, number of RX channels, number of TX/RXs).
        rf = reorderRf.process(rf);
        try
            set(hDisp, 'CData', gather(rf(:, selectedLines)));
            drawnow limitrate;
        catch ME
            if(strcmp(ME.identifier, 'MATLAB:class:InvalidHandle'))
                disp('Display was closed.');
            else
                rethrow(ME);
            end
        end
    end

catch e
    fprintf(1, 'Error: \n%s', e.identifier);
    fprintf(1, 'Message: \n%s', e.message);
    if ~isempty(session)
        % Please remember to close the session before re-running
        % the script!
        session.close();
        session = [];
    end
end

if ~isempty(session)
    % Please remember to close the session before re-running
    % the script!
    session.close();
    session = [];
end