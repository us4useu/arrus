function[img] = reconstructBModeImageLIN(rf,sys,seq,xGrid,zGrid)
% Output:
% img - resonstructed image
% 
% Input:
% rf - rf echo signal
% sys - system parameters structure
% seq - sequence parameters structure
% xGrid - image horizontal grid [m]
% zGrid - image vertical grid [m]

%% Parameters
% GPU availability
gpuEnable = license('test', 'Distrib_Computing_Toolbox') && ...
            ~isempty(ver('distcomp')) && parallel.gpu.GPUDevice.isAvailable;

% BP filter coefficients
[filtB,filtA] = butter(2,[0.5 1.5]*seq.txFreq/(seq.rxSampFreq/2),'bandpass');

% reconstruction (rec) parameters
rec.xGrid = xGrid;
rec.zGrid = zGrid;
rec.xSize = length(xGrid);
rec.zSize = length(zGrid);

rec.cicOrd = 2;
rec.dec = 4;
rec.iqEnable = true;

%% Processing
rf = double(rf);

% Move to GPU if possible
if gpuEnable
    rf = gpuArray(rf);
end

% Band pass filtration
rf = filter(filtB,filtA,rf);

% Down conversion
rf = downConversion(rf,seq,rec);

% Delay-and-Sum reconstruction
iqImg = reconstructRfLin(rf,sys,seq,rec);
            
% Envelope detection
img = abs(iqImg);

% Scan conversion
img = scanConversion(img,sys,seq,rec);

% Compression
img = 20*log10(img);

% Gather data from GPU
if gpuEnable
    img = gather(img);
end

end

