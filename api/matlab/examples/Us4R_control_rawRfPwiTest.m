
addpath('..\');
addpath('..\arrus');

us  = Us4R('configFile', 'us4r.prototxt');

% seq = CustomTxRxSequence(...
%                         'txApertureCenter', 0e-3, ...
%                         'txApertureSize',   us.getNProbeElem, ...
%                         'rxApertureCenter', 0e-3, ...
%                         'rxApertureSize',   us.getNProbeElem, ...
%                         'txFocus',          inf, ...
%                         'txAngle',          0, ...
%                         'speedOfSound',     1540, ...
%                         'txFrequency',      6.5e6, ...
%                         'txNPeriods',       2, ...
%                         'txVoltage',        20, ...
%                         'rxNSamples',       4*1024, ...
%                         'hwDdcEnable',      false, ...
%                         'txPri',            1e-3 ...
%                         );


% [lvl, dur] = pulse(0.2e6, 2);
% [lvl, dur] = chirpStep((1.00 : 0.5 : 5.00)*1e6);
[lvl, dur] = chirpCont([1 5]*1e6, 4e-6, [], 'matlab');

figure;
plotPulse(lvl,dur,'LineWidth',2,'Color','b');
grid on;

waveformBuilder = arrus.ops.us4r.WaveformBuilder();
wf = waveformBuilder.add(dur, lvl, 1).build();

seq = CustomTxRxSequence(...
                        'txApertureCenter', 0e-3, ...
                        'txApertureSize',   us.getNProbeElem, ...
                        'rxApertureCenter', 0e-3, ...
                        'rxApertureSize',   us.getNProbeElem, ...
                        'txFocus',          inf, ...
                        'txAngle',          0, ...
                        'speedOfSound',     1540, ...
                        'txWaveform',       wf, ...
                        'txVoltage',        20, ...
                        'rxNSamples',       4*1024, ...
                        'hwDdcEnable',      false, ...
                        'txPri',            1e-3 ...
                        );

us.upload(seq);
us.imageRawRf('amplitudeLim',1e3);
us.closeSession;

%% Subfunctions
function [lvl,dur] = pulse(freq,nCyc,fclk)
    % Generates TX signal: standard pulse
    % 
    % Inputs:
    % freq - center frequency [Hz]
    % nCyc - number of cycles
    % fclk - clock frequency [Hz] (optional)
    % 
    % Outputs:
    % lvl - TX levels
    % dur - TX levels durations

    if nargin<3 || isempty(fclk), fclk = 130e6; end
    dt = 1/fclk;

    durTot = round((1/freq)/(2*dt))*(2*dt);
    
    lvl = repmat([1, -1], 1, nCyc);
    dur = repmat(durTot/2, 1, 2*nCyc);
end

function [lvl,dur] = chirpStep(freq,fclk)
    % Generates TX signal: stepped chirp
    % 
    % Inputs:
    % freq - frequency of each chirp step [Hz]
    % fclk - clock frequency [Hz] (optional)
    % 
    % Outputs:
    % lvl - TX levels
    % dur - TX levels durations

    if nargin<2 || isempty(fclk), fclk = 130e6; end
    dt = 1/fclk;
    
    durCyc = round((1./freq)/(2*dt))*(2*dt);
    nCyc = numel(freq);
    
    lvl = repmat([1, -1], 1, nCyc);
    dur = reshape(durCyc/2 .* [1; 1], [], 1).';
end

function [lvl,dur] = chirpCont(freqLim,durTot,fclk,mode)
    % Generates TX signal: continuous chirp
    % 
    % Inputs:
    % freqLim - frequency limits [Hz]
    % durTot - total duration [s]
    % fclk - clock frequency [Hz] (optional)
    % mode - "raw" for raw implementation; "fixed" for forcing the frequency of the first and last half-cycles 
    % to correspond to freqLim; "matlab" for using matlab chirp function, result same as for "raw"; (optional)
    % 
    % Outputs:
    % lvl - TX levels
    % dur - TX levels durations

    if nargin<3 || isempty(fclk), fclk = 130e6; end
    if nargin<4 || isempty(mode), mode = "matlab"; end
    dt = 1/fclk;

    time = 0 : dt : durTot;
    nSamp = numel(time);
    
    switch mode
        case "raw"
            freq = linspace(freqLim(1), freqLim(2), nSamp);
            
            phs = cumsum(2*pi*freq*dt);
            amp = sin(phs);
            
        case "fixed"
            nSampBeg = round(1/freqLim(1)/dt);
            nSampEnd = round(1/freqLim(2)/dt);
            nSampMid = nSamp - nSampBeg - nSampEnd;
            
            freq = [freqLim(1).*ones(1,nSampBeg), linspace(freqLim(1), freqLim(2), nSampMid), freqLim(2).*ones(1,nSampEnd)];
            
            phs = cumsum(2*pi*freq*dt);
            amp = sin(phs);
            
        case "matlab"
            amp = chirp(time,freqLim(1),durTot,freqLim(2),"linear",270 + freqLim(1)*dt*360);
            
        otherwise
            error('Unsupported mode value, should be "raw", "fixed", or "matlab"');
    end

    sgn = sign(amp);
    
    aChngPos = find(sign(diff([0,sgn,0])));
    
    dur = diff(aChngPos)*dt;
    lvl = sgn(aChngPos(1:(end-1)));
    
    if (diff(freqLim) < 0 && dur(end) / dur(end-1) < 1) || ...
       (diff(freqLim) > 0 && dur(end) / dur(end-1) > 1)
        dur(end) = [];
        lvl(end) = [];
    end
end

function [] = plotPulse(lvl,dur,varargin)

t = reshape([0, cumsum(dur)] .* [1; 1], [], 1).';
a = [0, reshape(lvl .* [1; 1], [], 1).', 0];

plot(t, a, varargin{:});

end
