classdef Us4R < handle

    properties(GetAccess = protected, SetAccess = immutable, Transient = true, Hidden = true)
        ptr (1, 1) arrus.Ptr
    end
    methods
        function obj = Session(ptr)
            % Us4R handle constructor.
            %
            % :param ptr: pointer to the underlying device
        obj.ptr = arrus.Ptr("arrus.devices.us4r.Us4R", ptr);
        end

        function setVoltage(voltage)
             % Sets the voltage to given value.
             %
             % :param voltage: value to set
            obj.ptr.callMethod("setVoltage", voltage);
        end
    end
end