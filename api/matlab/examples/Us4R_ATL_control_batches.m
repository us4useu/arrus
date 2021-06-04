
% path to the MATLAB API files
addpath('../arrus');
addpath('C:\Users\Public\us4oem-releases\ref-US4R-21\matlab\')

txFrequency = 5e6;
samplingFrequency = 65e6;

[filtB,filtA] = butter(2,[0.5 1.5]*txFrequency/(samplingFrequency/2),'bandpass');

nUs4OEM = 2;

%% Initialize the system, sequence, and reconstruction
us	= Us4R('nUs4OEM',      nUs4OEM, ...
           'probeName',   'L7-4', ...
           'adapterType', 'atl/philips', ...
           'voltage',      50, ...
           'logTime',      true);
       
seqPWI = PWISequence(	'txApertureCenter', 0*1e-3, ...
                        'txApertureSize',   128, ...
                        'txFocus',          inf*1e-3, ...
                        'txAngle',          linspace(-12, 12, 17)*pi/180, ...
                        'speedOfSound',     1540, ...
                        'txFrequency',      txFrequency, ...
                        'txNPeriods',       2, ...
                        'rxNSamples',       1*1024, ...
                        'nRepetitions',     100, ...
                        'txPri',            59*1e-6, ...
                        'tgcStart',         14, ...
                        'tgcSlope',         2e2);

xGrid = (-20:0.05:20)*1e-3;
zGrid = (  0:0.05:40)*1e-3;
% GPU/CPU reconstruction implemented in matlab.
rec = Reconstruction(   'filterEnable',     true, ...
                        'filterACoeff',     filtA, ...
                        'filterBCoeff',     filtB, ...
                        'iqEnable',         true, ...
                        'cicOrder',         2, ...
                        'decimation',       4, ...
                        'xGrid',            (-20:0.05:20)*1e-3, ...
                        'zGrid',            (  0:0.05:40)*1e-3);

                    
                    % TODO check target PRI
us.upload(seqPWI,rec);

us.prepareBuffer(27);
us.acquireToBuffer(27);

triggerNumbers = [];
timestamps = [];

for i=1:25
    disp(['frame ', num2str(i)]);
    rf = us.popBufferElement();
    frameMetadata = rf(1, 1:1024*17:end);
    % trigger number
    % disp(frameMetadata);
    triggerNumbers = [triggerNumbers frameMetadata];
    % frame timestamp - the time when the frame was actually acquired.
    % metadataInt8 = typecast(frameMetadata, 'int8');
    % timestamp = metadataInt8(9:16); % bytes 8-16 contains timestamp
    % timestamp = double(typecast(timestamp, 'uint64'))/65e6;
    % timestamps = [timestamps timestamp];
    % disp(size(rf));
    % imagesc(rf);
%     rf = reshape(permute(reshape(rf,32,1024,2,17,100,2),[2 1 6 3 4 5]),1024,128,17,100);
%     obj.hFig = figure();
%     obj.hImg = imagesc(xGrid*1e3, zGrid*1e3,[]);
%     colormap(gray);
%     colorbar; 
%     daspect([1 1 1]);
%     set(gca,'CLim',[20 80]);
% 
%     for j = 1:100
%         img = us.reconstructOffline(rf(:,:,:,j));
%         set(obj.hImg, 'CData', img);
%         pause(0.1);
%     end
end


