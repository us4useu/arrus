classdef MatchingLayer
    % The matching layer applied directly on the probe elements.
    %
    % :param thickness: matching layer thickness [m]
    % :param speedOfSound: matching layer speed of sound [m/s]

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