
nSamples = 1024;
nAngles = 17;
nRepetitions = 100;
nBatches = 5;

xGrid = (-20:0.10:20)*1e-3;
zGrid = (  0:0.10:50)*1e-3;

% path to the MATLAB API files
addpath('../arrus');
addpath('C:\Users\Public\us4oem-releases\develop\matlab\')

txFrequency = 5e6;
samplingFrequency = 65e6;

[filtB,filtA] = butter(2,[0.5 1.5]*txFrequency/(samplingFrequency/2),'bandpass');

nUs4OEM = 2;

%% Initialize the system, sequence, and reconstruction
us	= Us4R('nUs4OEM',      nUs4OEM, ...
           'probeName',   'L7-4', ...
           'adapterType', 'atl/philips', ...
           'voltage',      40, ...
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
                        'cicOrder',         2, ...
                        'decimation',       4, ...
                        'xGrid',            xGrid, ...
                        'zGrid',            zGrid);

                    
                    % TODO check target PRI
us.upload(seqPWI,rec);

us.prepareBuffer(nBatches);
us.acquireToBuffer(nBatches);

triggerNumbers = [];
timestamps = [];

for i=1:1
    rf = us.popBufferElement();
    frameMetadata = rf(:, 1);
    % trigger number
    triggerNumber = frameMetadata(1);
    triggerNumbers = [triggerNumbers triggerNumber];
    % frame timestamp - the time when the frame was actually acquired.
    metadataInt8 = typecast(frameMetadata, 'int8');
    timestamp = metadataInt8(9:16); % bytes 8-16 contains timestamp
    timestamp = double(typecast(timestamp, 'uint64'))/65e6;
    timestamps = [timestamps timestamp];
    % disp(size(rf));
    % imagesc(rf);
end

rf = reshape(permute(reshape(rf,32,nSamples,2,nAngles,nRepetitions,2),[2 1 6 3 4 5]),nSamples,128,nAngles,nRepetitions);
img = us.reconstructOffline(rf);
figure, imagesc(xGrid,zGrid,img), colormap(gray), colorbar, daspect([1 1 1]), set(gca,'CLim',[20 80]);


