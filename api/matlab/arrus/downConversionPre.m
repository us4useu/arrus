% Precalculation of Digital Down Converter parameters
function[ddcPre] = downConversionPre(acq,proc)
% Outputs:
% 
% ddcPre                    - precalculated digit. down conv. (DDC) parameters
% ddcPre.demodSig           - demodulating signal
% ddcPre.cicFir             - FIR filter (equivalent to CIC filter)
% ddcPre.dec                - decimation factor
% 
% Inputs:
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
    nSamp = single(acq.nSamp);
    if proc.gpuEnable
        nSamp = gpuArray(nSamp);
    end
    
    t = (0:nSamp-1)'/acq.rxSampFreq;
    demodSig = 2.*exp(-1i*2*pi*acq.txFreq*t);
    demodSig = repmat(demodSig,1,acq.rxApSize,acq.nTx,acq.nRep);
else
    demodSig = 1;
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
if proc.gpuEnable
    cicFir	= gpuArray(cicFir);
    cicFir1 = gpuArray(cicFir1);
end
for ord=1:proc.cicOrd
    cicFir = conv(cicFir,cicFir1,'full');
end

%% Save the precalculated DDC parameters
ddcPre.demodSig	= demodSig;
ddcPre.cicFir	= cicFir;
ddcPre.dec      = proc.dec;

end