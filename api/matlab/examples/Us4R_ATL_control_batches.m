%% THIS MUST REMAIN UNCHANGED
addpath('../arrus');    % path to the MATLAB API files
nUs4OEM = 2;

%% PARAMETERS
% Acquisition parameters
nSamples = 1024;
nAngles = 17;
nRepetitions = 100;
nBatches = 100;

txFrequency = 15e6;
samplingFrequency = 65e6;

% Imaging parameters
reconstrEnable = false;

xGrid = (-20:0.10:20)*1e-3;
zGrid = (  0:0.10:50)*1e-3;

[filtB,filtA] = butter(2,[0.5 1.5]*txFrequency/(samplingFrequency/2),'bandpass');

%% Initialize the system, sequence, reconstruction, and data buffer
us	= Us4R('nUs4OEM',      nUs4OEM, ...
           'probeName',   'LA/20/128', ...
           'adapterType', 'atl/philips', ...
           'voltage',      10, ...
           'logTime',      true);
       
seqPWI = PWISequence(	'txApertureCenter', 0*1e-3, ...
                        'txApertureSize',   128, ...
                        'txFocus',          inf*1e-3, ...
                        'txAngle',          linspace(-12, 12, nAngles)*pi/180, ...
                        'speedOfSound',     1540, ...
                        'txFrequency',      txFrequency, ...
                        'txNPeriods',       2, ...
                        'rxNSamples',       nSamples, ...
                        'nRepetitions',     nRepetitions, ...
                        'txPri',            59*1e-6, ...
                        'tgcStart',         14, ...
                        'tgcSlope',         2e2);

% GPU/CPU reconstruction implemented in matlab.
rec = Reconstruction(   'filterEnable',     true, ...
                        'filterACoeff',     filtA, ...
                        'filterBCoeff',     filtB, ...
                        'iqEnable',         true, ...
                        'cicOrder',         1, ...
                        'decimation',       1, ...
                        'xGrid',            xGrid, ...
                        'zGrid',            zGrid);

                    
us.upload(seqPWI,rec);

us.prepareBuffer(nBatches);

%% acquire data and transfer to buffer
us.acquireToBuffer(nBatches);

%% load data from buffer to Matlab workspace & reorganize them
rf = zeros(nSamples,128,nAngles,nRepetitions,nBatches,'int16');
for iBatch=1:nBatches
    rf(:,:,:,:,iBatch) = getOldestBatchFromBuffer(us, nSamples, nAngles, nRepetitions);
end

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
    set(gca,'CLim',[20 80]);
end

%% --------------- SUBFUNCTIONS ---------------

function [rf] = getOldestBatchFromBuffer(usObj, nSamples, nAngles, nRepetitions)

rf = usObj.popBufferElement();

rf = reshape(rf,32,nSamples,2,nAngles,nRepetitions,2);
rf = permute(rf,[2 1 6 3 4 5]);
rf = reshape(rf,nSamples,128,nAngles,nRepetitions);

end