% path to the MATLAB API files
addpath('../arrus');

% parameters
nUs4OEM = 2;
probeName = 'SL1543';

%% define sequence

sequence = STASequence('txApertureCenter', 1:192, ...
                       'txApertureSize',   1, ...
                       'txFocus',          0*1e-3, ...
                       'txAngle',          0*pi/180, ...
                       'speedOfSound',     1450, ...
                       'txFrequency',      7e6, ...
                       'txNPeriods',       2, ...
                       'rxNSamples',       2^13, ...
                       'nRepetitions',     1, ...
                       'txPri',            200*1e-6, ...
                       'tgcStart',         14, ...
                       'tgcSlope',         2e2, ...
                       'fsDivider',        1);                        


%% run
us = Us4R(nUs4OEM, probeName, 50, true);
us.upload(sequence);
rf = us.run;
                    
%% test acquired data
% rf = ones(128);
testRfSize(rf)


%% test functions definitions

function testRfSize(rf)
    obtained = size(rf);
    expected = [8*1024, 192, 192];
    assert(isequal(obtained, expected))
end

