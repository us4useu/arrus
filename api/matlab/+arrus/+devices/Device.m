classdef Device < handle
    % Generic handle to device.

    properties(GetAccess = protected, SetAccess = immutable, Transient = true, Hidden = true)
        ptr (1, 1) arrus.Ptr
    end
    methods
        function obj = Device(ptr)
            % Us4R handle constructor.
            %
            % :param ptr: pointer to the underlying device
        obj.ptr = arrus.Ptr("arrus.devices.Device", ptr);
        end
    end
end