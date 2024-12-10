classdef WaveformBuilder
    % Tx waveform builder.

    properties(Access = private)
        segments arrus.ops.us4r.WaveformSegment = arrus.ops.us4r.WaveformSegment.empty;
        nRepeats;
    end

    methods
        function obj = WaveformBuilder()
        end

        function obj = add(obj, duration, state, nRepeats)
            % Adds segment (duration, state) with the given number of repeats.
            segment = arrus.ops.us4r.WaveformSegment('duration', duration, 'state', state);
            obj.segments(end+1) = segment;
            obj.nRepeats(end+1) = nRepeats;
        end

        function result = build(obj)
            % Builds the waveform.
            result = arrus.ops.us4r.Waveform('segments', obj.segments, 'nRepeats', obj.nRepeats);
        end
    end
end

