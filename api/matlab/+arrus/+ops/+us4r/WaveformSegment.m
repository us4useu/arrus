classdef WaveformSegment
    % Tx waveform segment.
    %
    %  NOTE: duration and state should have the same length!
    %
    % :param duration: duration of a given state; duration(i) is a duration of state(i) [seconds]
    % :param state: state to apply, available states: -2, -1, 0, 1, 2

    properties(Constant, Hidden=true)
        REQUIRED_PARAMS = {'duration', 'state'};
    end

    properties
        duration (1, :) {mustBePositive, mustBeReal}
        state (1, :) {mustBeInteger}
    end

    methods
        function obj = WaveformSegment(varargin)
            obj = arrus.utils.setArgs(obj, varargin, obj.REQUIRED_PARAMS);
        end

        function duration = getTotalDuration(obj)
            duration = sum(obj.duration);
        end
    end
end
