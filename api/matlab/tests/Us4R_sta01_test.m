% path to the MATLAB API files
addpath('../arrus');

% parameters
nUs4OEM = 2;
probeName = 'SL1543';
nChannels = 192;

txFrequency = 7e6;
samplingFrequency = 65e6;
fsDivider = 1;

[filtB,filtA] = butter(2,[0.5 1.5]*txFrequency/(samplingFrequency/fsDivider/2),'bandpass');

%% define sequence

sequence = STASequence('txApertureCenter', linspace(-15,15,nChannels)*1e-3, ...
                       'txApertureSize',   1, ...
                       'txFocus',          0*1e-3, ...
                       'txAngle',          0*pi/180, ...
                       'speedOfSound',     1450, ...
                       'txFrequency',      txFrequency, ...
                       'txNPeriods',       2, ...
                       'rxNSamples',       2^13, ...
                       'nRepetitions',     1, ...
                       'txPri',            200*1e-6, ...
                       'tgcStart',         14, ...
                       'tgcSlope',         0, ...
                       'fsDivider',        fsDivider);                        

                   
rec = Reconstruction(   'filterEnable',     true, ...
                        'filterACoeff',     filtA, ...
                        'filterBCoeff',     filtB, ...
                        'iqEnable',         true, ...
                        'cicOrder',         2, ...
                        'decimation',       1, ...
                        'xGrid',            (-20:0.10:20)*1e-3, ...
                        'zGrid',            (  0:0.10:50)*1e-3);                   

%% run
us = Us4R(nUs4OEM, probeName, 50, true);
% us.upload(sequence,rec);
% [rf,img] = us.run;

us.upload(sequence);
[rf] = us.run;
                    
%% test acquired data
% rf = ones(128);
testRfSize(rf, nChannels, sequence)
testTxPeak(rf, 124)
disp('All tests ok.')

%% test functions definitions

function testRfSize(rf, nChannels, sequence)
% test size of output rf array 

    obtained = size(rf);
    nSamp = sequence.rxNSamples(2) - sequence.rxNSamples(1) + 1;
    if isempty(sequence.txCenterElement)
        nTx = length(sequence.txApertureCenter);
    elseif isempty(sequence.txApertureCenter)
        nTx = length(sequence.txCenterElement);
    else
       error('There is no txApertureCenter, nor txCenterElement field in the sequence.') 
    end
    expected = [nSamp , nChannels, nTx];
    assert(isequal(obtained, expected))
end

function testTxPeak(rf, expectedPeakIndex)
% Test if peaks which comes from transmission is with expected delay.
% Eexpectations comes from prior measurements.

    for iRep = 1:size(rf,4)
        for iTx = 1:size(rf,3)
            for iRx = 1:size(rf,2)
                thisRf = rf(:,iRx, iTx, iRep);
                [~, imax] = max(thisRf);
                assert(expectedPeakIndex == imax)
            end
        end
    end
end
