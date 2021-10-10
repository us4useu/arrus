
%% PARAMETERS
nSamples = 1024;
nAngles = 17;
nRepetitions = 100;
nBatches = 100;

txVoltage = 10;
txFrequency = 15e6;
txAngles = linspace(-12, 12, nAngles)*pi/180;
txPri = 59e-6;

% Reconstruction parameters
reconstrEnable = false;
xGrid = (-7:0.05: 7)*1e-3;
zGrid = ( 0:0.05:12)*1e-3;

%% Initialize the system, sequence, reconstruction, and data buffer
us = initializeSystem(txVoltage,txFrequency,txAngles,nSamples,nRepetitions,txPri,xGrid,zGrid);
us.prepareBuffer(nBatches);

%% acquire data and transfer to buffer
us.acquireToBuffer(nBatches);

%% load data from buffer to Matlab workspace & reorganize them
rf = zeros(nSamples,128,nAngles,nRepetitions,nBatches,'int16');

wb = waitbar(0, 'Data import');
for iBatch=1:nBatches
    rf(:,:,:,:,iBatch) = getOldestBatchFromBuffer(us, nSamples, nAngles, nRepetitions);
    
    waitbar(iBatch/nBatches, wb);
end
close(wb);

%% reconstruct images offline
if reconstrEnable
    img = zeros(numel(zGrid), numel(xGrid), nRepetitions, nBatches);
    wb = waitbar(0, 'Reconstruction');
    for iBatch=1:nBatches
        for iRepetition=1:nRepetitions
            img(:,:,iRepetition,iBatch) = us.reconstructOffline(rf(:,:,:,iRepetition,iBatch));
        end
        waitbar(iBatch/nBatches, wb);
    end
    close(wb);
    
    figure;
    imagesc(xGrid,zGrid,img(:,:,1,1));
    colormap(gray);
    colorbar;
    daspect([1 1 1]);
    set(gca,'CLim',[0 80]);
end

%% --------------- SUBFUNCTIONS ---------------

function [usObj] = initializeSystem(txVoltage,txFrequency,txAngles,nSamples,nRepetitions,txPri,xGrid,zGrid)

%% System setup
addpath('../arrus');    % path to the MATLAB API files

usObj = Us4R(           'probeName',        'LA/20/128', ...
                        'adapterType',      'atl/philips', ...
                        'voltage',          txVoltage, ...
                        'logTime',          true);

%% Acquisition setup
seqPWI = PWISequence(	'txApertureCenter', 0*1e-3, ...
                        'txApertureSize',   128, ...
                        'txFocus',          inf*1e-3, ...
                        'txAngle',          txAngles, ...
                        'speedOfSound',     1540, ...
                        'txFrequency',      txFrequency, ...
                        'txNPeriods',       2, ...
                        'rxNSamples',       nSamples, ...
                        'nRepetitions',     nRepetitions, ...
                        'txPri',            txPri, ...
                        'tgcStart',         14, ...
                        'tgcSlope',         2e2);

%% Reconstruction setup
if nargin==8 && ~isempty(xGrid) && ~isempty(zGrid)
    
    samplingFrequency = 65e6;
    [filtB,filtA] = butter(2,[0.5 1.5]*txFrequency/(samplingFrequency/2),'bandpass');   % Band pass filter coefficients
    
    rec = Reconstruction(   'filterEnable',     true, ...
                            'filterACoeff',     filtA, ...
                            'filterBCoeff',     filtB, ...
                            'iqEnable',         true, ...
                            'cicOrder',         1, ...
                            'decimation',       1, ...
                            'xGrid',            xGrid, ...
                            'zGrid',            zGrid);
end

%% Prepare the acquisition (& reconstruction)
usObj.upload(seqPWI,rec);

end

function [rf] = getOldestBatchFromBuffer(usObj, nSamples, nAngles, nRepetitions)

rf = usObj.popBufferElement();

rf = reshape(rf,32,nSamples,2,nAngles,nRepetitions,2);
rf = permute(rf,[2 1 6 3 4 5]);
rf = reshape(rf,nSamples,128,nAngles,nRepetitions);

end


