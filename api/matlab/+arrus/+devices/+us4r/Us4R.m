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

        function setTgcCurve(varargin)
            %
            % Sets TGC curve points asynchronously.
            % Setting empty vectors t and y turns off analog TGC. Setting non-empty vector turns off DTGC
            %  and turns on analog TGC.
            % Vectors t and y should have exactly the same size. The input t and y values will be interpolated
            % into target hardware sampling points (according to getCurrentSamplingFrequency and getCurrentTgcPoints).
            % Linear interpolation will be performed, the TGC curve will be extrapolated with the first
            % (left-side of the cure) and the last sample (right side of the curve).
            %
            % :param time: sampling time, relative to the "sample 0" (optional, hardware sampling time will be used
            % if not provided)
            % :param value: values to apply at given sampling time
            % :param applyCharacteristic: set it to true if you want to compensate response characteristic
            % (pre-computed by us4us).
            obj = varargin{1};
            if nargin == 4
                time = varargin{2};
                value = varargin{3};
                applyCharacteristic = varargin{4};
                obj.ptr.callMethod("setTgcCurveTimeValue", 0, time, value, logical(applyCharacteristic));
            elseif nargin == 3
                value = varargin{2};
                applyCharacteristic = varargin{3};
                obj.ptr.callMethod("setTgcCurveValue", 0, value, logical(applyCharacteristic));
            else
                error("Unsupported number of parameters.");
            end
        end

        function [gain] = getLnaGain(obj)
            %
            % Returns current LNA gain value.
            %
            % :return: LNA gain value [dB]
            res = obj.ptr.callMethod("getLnaGain", 1);
            gain = res{1, 1};
        end

        function setLnaGain(obj, gain)
            %
            % Sets the LNA gain to the given value.
            %
            % :param gain: gain value to set [dB]
            obj.ptr.callMethod("setLnaGain", 0, gain);
        end

        function [gain] = getPgaGain(obj)
            %
            % Returns current PGA gain value.
            %
            % :return: PGA gain value [dB]
            res = obj.ptr.callMethod("getPgaGain", 1);
            gain = res{1, 1};
        end

        function setPgaGain(obj, gain)
            %
            % Sets the PGA gain to the given value.
            %
            % :param gain: gain value to set [dB]
            obj.ptr.callMethod("setPgaGain", 0, gain);
        end

        function disableLnaHpf(obj)
            %
            % Disables LNA analog high pass filter.
            %
            obj.ptr.callMethod("disableLnaHpf", 0);
        end

        function setLnaHpfCornerFrequency(obj, frequency)
            %
            % Enables the LNA analog high pass filter and sets its corner frequency to the given value.
            %
            % :param frequency: corner frequency to set [Hz]
            obj.ptr.callMethod("setLnaHpfCornerFrequency", 0, frequency);
        end

        function disableAdcHpf(obj)
            %
            % Disables ADC digital high pass filter.
            %
            obj.ptr.callMethod("disableAdcHpf", 0);
        end

        function setAdcHpfCornerFrequency(obj, frequency)
            %
            % Enables the ADC digital high pass filter and sets its corner frequency to the given value.
            %
            % :param frequency: corner frequency to set [Hz]
            obj.ptr.callMethod("setAdcHpfCornerFrequency", 0, frequency);
        end
    end
end