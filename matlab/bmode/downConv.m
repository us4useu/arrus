%Digital Down Converter - demodulacja kwadraturowa, decymacja + CIC/IIR
function[iq] = downConv(rf,sys,dec,cicOrd)

nSamp = size(rf,1);

%% Quadrature demodulation
t = (0:nSamp-1)'/sys.fs;

iq = 2*rf.*cos(-2*pi*sys.fn*t) ...
    +2*rf.*sin(-2*pi*sys.fn*t)*1i;

%% Downsampling (CIC filtration + decimation)
% Integrator
for ord=1:cicOrd
    iq = cumsum(iq);
end

% Decimator
iq = iq(dec:dec:end,:,:);

% Comb
for ord=1:cicOrd
    iq = [iq(1,:,:); diff(iq)];
end

end