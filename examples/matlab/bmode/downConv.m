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
% acq.rx.samplingFrequency	- [Hz] sampling frequency
% acq.tx.frequency          - [Hz] carrier (nominal) frequency
% 
% proc                      - processing-related parameters
% proc.ddc.iqEnable         - [logical] enables the quadrature demodulation
% proc.ddc.cicOrder         - [] order of the CIC filter
% proc.ddc.decimation       - [] decimation factor

%% Quadrature demodulation
if proc.ddc.iqEnable
    nSample = size(rfIn,1);
    t = (0:nSample-1)'/acq.rx.samplingFrequency;
    
    rfOut = 2*rfIn.*cos(-2*pi*acq.tx.frequency*t) ...
          + 2*rfIn.*sin(-2*pi*acq.tx.frequency*t)*1i;
else
    rfOut = rfIn;
end

%% Downsampling (CIC filtration + decimation)
% Integrator
for ord=1:proc.ddc.cicOrder
    rfOut = cumsum(rfOut);
end

% Decimator
rfOut = rfOut(proc.ddc.decimation:proc.ddc.decimation:end,:,:);

% Comb
for ord=1:proc.ddc.cicOrder
    rfOut = [rfOut(1,:,:); diff(rfOut)];
end

end