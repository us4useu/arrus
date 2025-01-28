classdef Pulse
    % A description of a transmitted pulse.
    %
    % :param centerFrequency: pulse transmit center frequency in [Hz]. Required.
    % :param nPeriods: number of sine burst periods, should be full and half cycles
    %   are supported (e.g. 1, 1.5, etc.) Required.
    % :param inverse: true if the polarity of the generated signal should be inverted,
    %   Optional, default value: false
    properties(Constant, Hidden=true)
        REQUIRED_PARAMS = {'centerFrequency', 'nPeriods'};
    end

    properties
        centerFrequency (1, 1) {mustBeFinite, mustBeReal, mustBePositive} = 1e6
        nPeriods (1, 1) {mustBeFinite, mustBeReal, mustBePositive} = 1
        inverse (1, 1) {arrus.validators.mustBeLogical} = false
        amplitudeLevel (1, 1) {mustBeMember(amplitudeLevel, {0, 1})} = 0
    end

    methods
        function obj = Pulse(varargin)
            obj = arrus.utils.setArgs(obj, varargin, obj.REQUIRED_PARAMS);
        end
    end

end
