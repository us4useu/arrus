% Digital Down Converter - quadrature demodulation, filtration & decimation
function[rfOut] = downConversion(rfIn,fn,fs,dec,ddcFirCoeff)
% Outputs:
% rfOut         - (nSamp/dec,nRx,nTx) output rf/iq data
% 
% Inputs:
% rfIn          - (nSamp,nRx,nTx) input rf data
% fn            - [Hz] carrier (nominal) frequency
% fs            - [Hz] sampling frequency
% dec           - [] decimation factor
% ddcFirCoeff   - [] FIR filter coefficients (full set)

%% Quadrature demodulation
nSamp = size(rfIn,1);
t = (0:nSamp-1)'/fs;

rfOut = 2*rfIn.*exp(-2i*pi*reshape(fn,1,1,[]).*t);

%% Downsampling (filtration + decimation)
rfOut = convn(rfOut,ddcFirCoeff(:),'same');
rfOut = rfOut(dec:dec:end,:,:,:);

end