% Digital Down Converter - quadrature demodulation, filtration & decimation
function[rfOut] = downConversion(rfIn,acq,proc)
% Outputs:
% 
% rfOut                     - (nSamp/dec,nRx,nTx) output rf/iq data
% 
% 
% Inputs:
% 
% rfIn                      - (nSamp,nRx,nTx,nRep) input rf data
% 
% acq                       - acquisition-related parameters
% acq.rxSampFreq            - [Hz] sampling frequency
% acq.txFreq                - [Hz] carrier (nominal) frequency
% 
% proc                      - processing-related parameters
% proc.ddcFirCoeff          - [] FIR filter coefficients (full set)
% proc.dec                  - [] decimation factor

%% Quadrature demodulation
[nSamp,nRx,nTx,nRep] = size(rfIn);
t = (0:nSamp-1)'/acq.rxSampFreq;

rfOut = 2*rfIn.*exp(-2i*pi*reshape(acq.txFreq,1,1,[]).*t);

%% Downsampling (filtration + decimation)
rfOut = conv2(rfOut(:,:),proc.ddcFirCoeff(:),'same');
rfOut = reshape(rfOut,nSamp,nRx,nTx,nRep);
rfOut = rfOut(proc.dec:proc.dec:end,:,:,:);

end