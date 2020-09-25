classdef Pulse
    % A description of a transmitted pulse.
    %
    % :param centerFrequency: pulse transmit center frequency in [Hz]
    % :param nPeriods: number of sine burst periods, should be full and half cycles
    %   are supported (e.g. 1, 1.5, etc.)
    % :param inverse: true if the polarity of the generated signal should be inverted,
    %   optional, default value: false

    properties
        centerFrequency
        nPeriods
        inverse
    end

    methods
        function obj = Pulse(varargin)
            p = inputParser;
            addRequired(p, 'centerFrequency', @(x) isscalar(x) && numeric(x) ...
                                                   && isfinite(x) && (x > 0) && isreal(x));
            addRequired(p, 'nPeriods', @(x) isscalar(x) && isinteger(x) && (x > 0));
            addOptional(p, 'inverse', false, @(x) isscalar(x) && islogical(x));
            parse(p,varargin{:})

            obj.centerFrequency = p.Results.frequency;
            obj.nPeriods = p.Results.nPeriods;
            obj.inverse = p.Results.inverse;
        end
    end
    
end
