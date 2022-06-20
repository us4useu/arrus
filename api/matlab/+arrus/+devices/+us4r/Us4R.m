classdef Us4R < handle

    properties(GetAccess = protected, SetAccess = immutable, Transient = true, Hidden = true)
        ptr arrus.Ptr {mustBeScalarOrEmpty}
    end
    methods
        function obj = Us4R(ptr)
            %
            % Us4R handle constructor.
            %
            % :param ptr: pointer to the underlying device
        obj.ptr = arrus.Ptr("arrus.devices.us4r.Us4R", ptr);
        end

        function setVoltage(obj, voltage)
             %
             % Sets the voltage to given value.
             %
             % :param voltage: value to set
            obj.ptr.callMethod("setVoltage", 0, voltage);
        end
    end
end