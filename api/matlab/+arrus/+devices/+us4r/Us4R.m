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

        function disableHV(obj)
            %
            % Disables HV.
            %
            obj.ptr.callMethod("disableHV", 0);
        end

        function setVoltage(obj, voltage)
            %
            % Sets the voltage to given value.
            %
            % :param voltage: value to set
            obj.ptr.callMethod("setVoltage", 0, voltage);
        end

        function setRDAC(obj, rdac)
            %
            % Sets the rdac to given value.
            %
            % :param rdac: value to set
            obj.ptr.callMethod("setRDAC", 0, rdac);
        end

        function [rdac] = getRDAC(obj)
            %
            % Returns the rdac value.
            %
            % :return: rdac value.
            res = obj.ptr.callMethod("getRDAC", 1);
            rdac = res{1, 1};
        end

        function [frequency] = getSamplingFrequency(obj)
            %
            % Returns NOMINAL sampling frequency.
            %
            % :return: nominal sampling frequency [Hz]
            res = obj.ptr.callMethod("getSamplingFrequency", 1);
            frequency = res{1, 1};
        end

        function [model] = getProbeModel(obj)
            %
            % Returns probe model definition.
            %
            % :return: probe model definition, an instance of class arrus.devices.probe.ProbeModel
            res = obj.ptr.callMethod("getProbeModel", 1);
            model = res{1, 1};
        end

        function [channelsMask] = getChannelsMask(obj)
            %
            % Returns list of masked elements/system channels.
            % Note: channel numbering starts from 0.
            %
            res = obj.ptr.callMethod("getChannelsMask", 1);
            channelsMask = res{1, 1};
        end
    end
end