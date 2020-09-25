classdef Tx
    % A 'transmit' pulse operation.
    %
    % Example usage:
    %
    % .. code:: matlab
    %
    %   aperture = false(128);
    %   aperture(2) = 1;
    %   aperture(4) = 1;
    %   pulse = ....
    %   Tx('aperture', aperture, 'delays', [5e-6, 6e-6], 'pulse', pulse);
    %
    % :param aperture: logical mask where 0 and 1 corresponds to active and inactive element respectively
    % :param delays: vector of transmit delays in [s], should be the size of the number of active channels in Tx
    %   aperture; delays[i] will be applied to the i-th active tx channel
    % :param pulse: a pulse to transmit, object of type `arrus.ops.us4r.Pulse`

    properties
        aperture
        delays
        pulse 
    end
    
    methods
        function obj = Tx(varargin)
            p = inputParser;
            addRequired(p, 'aperture', @(x) isvector(x) && islogical(x));
            addRequired(p, 'delays', @(x) isvector(x) && isreal(x));
            addRequired(p, 'pulse', @(x) isscalar(x) && isa(x, 'arrus.ops.us4r.Pulse'));
            parse(p,varargin{:});
            obj.aperture = p.Results.aperture;
            obj.delay = p.Results.delays;
            obj.pulse = p.Results.pulse;
        end
    end
    
end
