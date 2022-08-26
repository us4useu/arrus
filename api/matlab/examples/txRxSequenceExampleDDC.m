% Test script
clear all;
%addpath("/home/pjarosik/arrus-releases/ref-US4R-80/matlab/");
addpath("/home/pjarosik/src/arrus/build/api/matlab/wrappers");
addpath("/home/pjarosik/src/arrus/api/matlab/");
% addpath("C:\Users\pjarosik\src\arrus\api\matlab");

import arrus.ops.us4r.*;
import arrus.framework.*;

arrus.initialize("clogLevel", "INFO", "logFilePath", "arrus.log", "logFileLevel", "TRACE");

session = arrus.session.Session("/home/pjarosik/us4r.prototxt");
us4r = session.getDevice("/Us4R:0");

fn = 6e6;
nchannels = 192;

pulse = arrus.ops.us4r.Pulse('centerFrequency', fn, "nPeriods", 3.5);

seq = TxRxSequence( ...
    "ops", [ ...
      TxRx(...
        "tx", Tx("aperture", true(1, nchannels), 'delays', zeros(1, nchannels), "pulse", pulse), ...
        "rx", Rx("aperture", true(1, nchannels), "sampleRange", [0, 4096]), ...
        "pri", 200e-6), ...
       TxRx(...
         "tx", Tx("aperture", true(1, nchannels), 'delays', linspace(5e-6, 0, nchannels), "pulse", pulse), ...
         "rx", Rx("aperture", true(1, nchannels), "sampleRange", [0, 4096]), ...
         "pri", 200e-6)]);

cutoffFrequency = fn/(us4r.getSamplingFrequency()/2);
decimationFactor = 2;
filterOrder = decimationFactor * 16;
filterCoefficients = ones(1, filterOrder); %fir1(filterOrder, cutoffFrequency, "low"); requires signal processing toolbox
filterCoefficients = filterCoefficients(1, 1:length(filterCoefficients)/2);

disp(size(filterCoefficients));

digitalDownConversion = DigitalDownConversion( ...
    "demodulationFrequency", fn, ...
    "decimationFactor", decimationFactor, ...
    "firCoefficients", filterCoefficients ...
);
scheme = Scheme('txRxSequence', seq, 'workMode', "MANUAL", 'digitalDownConversion', digitalDownConversion);

us4r.setVoltage(5);
buffer = session.upload(scheme);
session.run();
array = buffer.front().eval();
disp("SIZE");
disp(size(array));

session.close();
% Reordering data (an implementation of this remapping step will be provided soon).
% ESAOTE 3
img = zeros(nchannels, 2, 4096, 2, 'int16');
[nRx, nComponents, nSamples, nTx] = size(img);
for i=1:nTx % TX
    for j=1:3 % RX subaperture number
        for u=1:2 % us4oem number
            chNumbers = (j-1)*64+(u-1)*32;
            frameNumber = nTx*3*(u-1) + (i-1)*3+j-1; % us4oem, TX, subaperture
            img((chNumbers+1):(chNumbers+32), :, 1:end, i) = array(:, :, (frameNumber*nSamples+1):((frameNumber+1)*nSamples));
        end
    end
end
img = permute(img, [2, 3, 1, 4]);
disp(size(img));

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
