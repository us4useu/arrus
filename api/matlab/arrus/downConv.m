% Digital Down Converter - quadrature demodulation, decimation & CIC filtration
function[rfOut] = downConv(rfIn,acq,proc)
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
% proc.iqEnable             - [logical] enables the quadrature demodulation
% proc.cicOrd               - [] order of the CIC filter
% proc.dec                  - [] decimation factor

%% Quadrature demodulation
if proc.iqEnable
    nSample = size(rfIn,1);
    t = (0:nSample-1)'/acq.rxSampFreq;
    
    rfOut = 2*rfIn.*cos(-2*pi*acq.txFreq*t) ...
          + 2*rfIn.*sin(-2*pi*acq.txFreq*t)*1i;
else
    rfOut = rfIn;
end

%% Downsampling (CIC filtration + decimation)
% Integrator
for ord=1:proc.cicOrd
    rfOut = cumsum(rfOut);
end

% Decimator
rfOut = rfOut(proc.dec:proc.dec:end,:,:);

% Comb
for ord=1:proc.cicOrd
    rfOut = [rfOut(1,:,:); diff(rfOut)];
end

end