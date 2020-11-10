% Digital Down Converter - quadrature demodulation, decimation & CIC filtration
function[rfOut] = downConversion(rfIn,ddcPre)
% Outputs:
% 
% rfOut                     - (nSamp/dec,nRx,nTx) output rf/iq data
% 
% Inputs:
% 
% rfIn                      - (nSamp,nRx,nTx) input rf data
% 
% ddcPre                    - precalculated digit. down conv. (DDC) parameters
% ddcPre.demodSig           - demodulating signal
% ddcPre.cicFir             - FIR filter (equivalent to CIC filter)
% ddcPre.dec                - decimation factor

%% Quadrature demodulation
rfOut = rfIn.*ddcPre.demodSig;

%% Downsampling (CIC filtration + decimation)
% Initially there was an ordinary CIC filter here. CIC is fast & simple in 
% hardware implementations. Here it needed for-loops and cumsum-function 
% (relatively slow). So, the present implementation is a FIR filter which:
% - gives same results
% - can run much faster (especially on GPU)

[nSamp,nRx,nTx,nRep] = size(rfIn);

% Filter & decimate
rfOut = reshape(rfOut,nSamp,nRx*nTx*nRep);
rfOut = conv2(rfOut,ddcPre.cicFir,'same');
rfOut = rfOut(ddcPre.dec:ddcPre.dec:end,:);
rfOut = reshape(rfOut,[],nRx,nTx,nRep);

end