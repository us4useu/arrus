% Test script
clear all;
addpath("/home/pjarosik/arrus-releases/ref-US4R-80/matlab/");
addpath("/home/pjarosik/src/arrus/build/api/matlab/wrappers");
% addpath("C:\Users\pjarosik\src\arrus\api\matlab");

% addpath("/home/pjarosik/src/arrus/api/matlab/");

import arrus.ops.us4r.*;
import arrus.framework.*;

arrus.initialize("clogLevel", "INFO", "logFilePath", "arrus.log", "logFileLevel", "TRACE");

pulse = arrus.ops.us4r.Pulse('centerFrequency', 5e6, "nPeriods", 3.5);

nchannels = 192;

seq = TxRxSequence( ...
    "ops", [ ...
      TxRx(...
        "tx", Tx("aperture", true(1, nchannels), 'delays', zeros(1, nchannels), "pulse", pulse), ...
        "rx", Rx("aperture", true(1, nchannels), "sampleRange", [0, 4096]), ...
        "pri", 110e-6), ...
       TxRx(...
         "tx", Tx("aperture", true(1, nchannels), 'delays', linspace(5e-6, 0, nchannels), "pulse", pulse), ...
         "rx", Rx("aperture", true(1, nchannels), "sampleRange", [0, 4096]), ...
         "pri", 110e-6)]);

scheme = Scheme('txRxSequence', seq, 'workMode', "MANUAL");

session = arrus.session.Session("/home/pjarosik/us4r.prototxt");
us4r = session.getDevice("/Us4R:0");
us4r.setVoltage(5);
buffer = session.upload(scheme);
session.run();
array = buffer.front().eval();

session.close();

% Reordering data (an implementation of this remapping step will be provided soon).
% ESAOTE 3
img = zeros(nchannels, 4096, 2, 'int16');
[nRx, nSamples, nTx] = size(img);
for i=1:nTx % TX
    for j=1:3 % RX subaperture number
        for u=1:2 % us4oem number
            chNumbers = (j-1)*64+(u-1)*32;
            frameNumber = nTx*3*(u-1) + (i-1)*3+j-1; % us4oem, TX, subaperture
            img((chNumbers+1):(chNumbers+32), 1:end, i) = array(:, (frameNumber*nSamples+1):((frameNumber+1)*nSamples));
        end
    end
end

% ATL/PHILIPS
% img = zeros(nchannels, 4096, 2, 'int16');
% [nRx, nSamples, nTx] = size(img);
% for i=1:nTx % TX
%     for j=1:2 % RX subaperture number
%         for u=1:2 % us4oem number
%             chNumbers = (j-1)*64+(u-1)*32;
%             frameNumber = nTx*2*(u-1) + (i-1)*2+j-1; % us4oem, TX, subaperture
%             img((chNumbers+1):(chNumbers+32), 1:end, i) = array(:, (frameNumber*nSamples+1):((frameNumber+1)*nSamples));
%         end
%     end
% end
% save("img.mat", "img");
