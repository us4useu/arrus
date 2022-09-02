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

        function [frequency] = getSamplingFrequency(obj)
            %
            % Returns NOMINAL sampling frequency.
            %
            % :return: nominal sampling frequency [Hz]
            res = obj.ptr.callMethod("getSamplingFrequency", 1);
            frequency = res{1, 1};
        end

        function enableAfeAutoOffsetRemoval(obj)
            %
            % Enables the AFE auto offset removal.
            %
            obj.ptr.callMethod("enableAfeAutoOffsetRemoval", 0);
        end

        function disableAfeAutoOffsetRemoval(obj)
            %
            % Disables the AFE auto offset removal.
            %
            obj.ptr.callMethod("disableAfeAutoOffsetRemoval", 0);
        end

        function setAfeAutoOffsetRemovalCycles(obj, nCyclesId)
            %
            % Sets the AFE auto offset removal accumulator cycles number identifier to given value.
            %
            % :param nCyclesId: value to set
            obj.ptr.callMethod("setAfeAutoOffsetRemovalCycles", 0, nCyclesId);
        end

        function setAfeAutoOffsetRemovalDelay(obj, delay)
            %
            % Sets the AFE auto offset removal accumulator delay to given value.
            %
            % :param delay: value to set
            obj.ptr.callMethod("setAfeAutoOffsetRemovalDelay", 0, delay);
        end

    end
end