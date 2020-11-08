% Digital Down Converter - quadrature demodulation, decimation & CIC filtration
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
% proc.iqEnable             - [logical] enables the quadrature demodulation
% proc.cicOrd               - [] order of the CIC filter
% proc.dec                  - [] decimation factor

%% Quadrature demodulation
if proc.iqEnable
    nSample = size(rfIn,1);
    nSample = single(nSample);
    if isa(rfIn,'gpuArray')
        nSample = gpuArray(nSample);
    end
    
    t = (0:nSample-1)'/acq.rxSampFreq;
    demodSignal = 2.*exp(-1i*2*pi*acq.txFreq*t);
    rfOut = rfIn.*demodSignal;
else
    rfOut = rfIn;
end

%% Downsampling (CIC filtration + decimation)
% Initially there was an ordinary CIC filter here. CIC is fast & simple in 
% hardware implementations. Here it needed for-loops and cumsum-function 
% (relatively slow). So, the present implementation is a FIR filter which:
% - gives same results
% - can run much faster (especially on GPU)

% Prepare the filter vector
cicFir	= single(1);
cicFir1	= ones(proc.dec,1,'single');
if isa(rfIn,'gpuArray')
    cicFir	= gpuArray(cicFir);
    cicFir1 = gpuArray(cicFir1);
end
for ord=1:proc.cicOrd
    cicFir = conv(cicFir,cicFir1,'full');
end

% Filter & decimate
rfOut = reshape(rfOut,acq.nSamp,acq.rxApSize*acq.nTx*acq.nRep);
rfOut = conv2(rfOut,cicFir,'same');
rfOut = rfOut(proc.dec:proc.dec:end,:);
rfOut = reshape(rfOut,[],acq.rxApSize,acq.nTx,acq.nRep);

end