% Digital Down Converter - quadrature demodulation, filtration & decimation
function[rfOut] = downConversion(rfIn,acq,proc)
% Outputs:
% 
% rfOut                     - (nSamp/dec,nRx,nTx) output rf/iq data
% 
% 
% Inputs:
% 
% rfIn                      - (nSamp,nRx,nTx) input rf data
% 
% acq                       - acquisition-related parameters
% acq.rxSampFreq            - [Hz] sampling frequency
% acq.txFreq                - [Hz] carrier (nominal) frequency
% 
% proc                      - processing-related parameters
% proc.ddcFirCoeff          - [] FIR filter coefficients (full set)
% proc.dec                  - [] decimation factor

%% Quadrature demodulation
nSample = size(rfIn,1);
t = (0:nSample-1)'/acq.rxSampFreq;

rfOut = 2*rfIn.*exp(-2i*pi*reshape(acq.txFreq,1,1,[]).*t);

%% Downsampling (filtration + decimation)
rfOut = convn(rfOut,proc.ddcFirCoeff(:),'same');
rfOut = rfOut(proc.dec:proc.dec:end,:,:,:);

end