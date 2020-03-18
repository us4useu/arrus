% PW Doppler
function[spect] = dopplerPW(iq,win,ovl)
% Pulsed Wave Doppler processing.
% 
% Outputs:
% spect         - (win,nWin)  output spectrum/a
% 
% Inputs:
% iq            - iq slow-time echo signal (vector)
% win           - [samples] FFT window length (optional; if undefined then win = nSamples)
% ovl           - [samples] FFT window overlap (optional; if undefined then ovl = 0)
% 
% If (due to win & ovl values) just one FFT window can be used (nWin==1), then the spect is a vector, 
% otherwise it is an array.

iq = squeeze(iq);
nSamples = length(iq);

if nargin<3
    ovl = 0;
end

if nargin<2
    win = nSamples;
end

sWinFirst = 1:(win-ovl):(nSamples-win+1);   % numbers of first samples in each window       
iqAux = iq(sWinFirst + (0:(win-1))');
spect = abs(fftshift(fft(iqAux),1));

end
