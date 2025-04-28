classdef Lens
    % The lens applied on the surface of the probe.
    %
    % Currently, the model of the lens is quite basic and accustomed mostly to
    % the linear array probes, e.g. we assume that the lens is dedicated to be
    % focusing in the elevation direction.

    % :param thickness: lens thickness measured at center of the elevation [m],
    % :param speedOfSound: the speed of sound in the lens material [m/s],
    % :param focus: geometric elevation focus in water [m]


    properties(Constant, Hidden=true)
        REQUIRED_PARAMS = {"thickness", "speedOfSound", "focus"};
    end

    properties
        thickness
        speedOfSound
        focus
    end

    methods
        function obj = Lens(varargin)
            obj = arrus.utils.setArgs(obj, varargin, obj.REQUIRED_PARAMS);
        end
    end

end