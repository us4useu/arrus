classdef Waveform
    % Tx waveform.
    %
    %  NOTE: segments and nRepeats should have the same length!
    %
    % :param segments: a list of waveform segments
    % :param nRepeats: how many times a given segments[i] should be repeated

    properties(Constant, Hidden=true)
        REQUIRED_PARAMS = {'segments', 'nRepeats'};
    end

    properties
        segments arrus.ops.us4r.WaveformSegment
        nRepeats (1, :) {mustBePositive, mustBeInteger}
    end

    methods
        function obj = Waveform(varargin)
            obj = arrus.utils.setArgs(obj, varargin, obj.REQUIRED_PARAMS);
        end

        function duration = getTotalDuration(obj)
            nSegments = length(obj.segments);
            duration = 0;
            for i=1:nSegments
                duration = duration + obj.segments(i).getTotalDuration()*obj.nRepeats(i);
            end
        end
        
    end

end
