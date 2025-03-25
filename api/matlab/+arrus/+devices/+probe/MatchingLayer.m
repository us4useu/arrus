classdef MatchingLayer
    % The matching layer between the lens and probe.
    %
    % :param thickness: lens thickness of linear array, measured at center of the elevation [m]
    % :param speedOfSound: the speed of sound in the lens material [m/s]

    properties(Constant, Hidden=true)
        REQUIRED_PARAMS = {"thickness", "speedOfSound"};
    end

    properties
        thickness
        speedOfSound
    end

    methods
        function obj = MatchingLayer(varargin)
            obj = arrus.utils.setArgs(obj, varargin, obj.REQUIRED_PARAMS);
        end
    end

end