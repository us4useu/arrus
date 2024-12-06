classdef Tx
    % Ultrasound pulse transmission.
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
    % :param pulse: a pulse to transmit, object of type `arrus.ops.us4r.Pulse` or `arrus.ops.us4r.Waveform`

    properties(Constant, Hidden=true)
        REQUIRED_PARAMS = {'aperture', 'delays', 'pulse'};
    end

    properties
        aperture (1, :) {arrus.validators.mustBeLogical}
        delays (1, :) {mustBeNonnegative, mustBeFinite, mustBeReal}
        pulse
    end

    methods
        function obj = Tx(varargin)
            obj = arrus.utils.setArgs(obj, varargin, obj.REQUIRED_PARAMS);
            mustBeA(obj.pulse, ["arrus.ops.us4r.Pulse", "arrus.ops.us4r.Waveform"]);
        end
    end
end
