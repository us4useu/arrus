classdef Session < handle
    % A communication session with a group of devices.

    properties(GetAccess = protected, SetAccess = immutable, Transient = true, Hidden = true)
        ptr arrus.Ptr {mustBeScalarOrEmpty}
    end
    methods
        function obj = Session(settings)
            % Session object constructor.
            %
            % :param sessionSettings: path to the session configuration file or arrus.session.SessionSettings instance
            obj.ptr = arrus.UniquePtr("arrus.session.Session", convertCharsToStrings(settings));
        end
        
        function res = getDevice(obj, deviceId)
            % Returns a device with given identifier
            %
            % :param deviceId: a string pointing to some device, e.g. to get first available us4R device, use "/Us4R:0"
            res = obj.ptr.callMethod("getDevice", deviceId);
        end

        function [buffer, metadata] = upload(obj, scheme)
            % Uploads a given scheme on the available devices.
            % Currently, the scheme upload is performed on the Us4R:0 device only.
            % After uploading a new sequence the previously returned output buffers will be in invalid state.
            %
            % :param scheme: scheme to upload (arrus.ops.us4r.Scheme)
            % :return: upload result information: output data buffer, metadata describing the data that will be generated
            [buffer, metadata] = obj.ptr.callMethod("upload", scheme);
        end

        function run()
            %
            % Runs the uploaded scheme.
            %
            % The behaviour of this method depends on the uploaded work mode:
            % - MANUAL: triggers execution of batch of sequences only ONCE,
            % - HOST, ASYNC: triggers execution of batch of sequences IN A LOOP (Host: trigger is on buffer element release).
            %   The run function can be called only once (before the scheme is stopped).
            obj.ptr.callMethod("run");
        end

        function startScheme()
            %
            % Starts the uploaded scheme.
            %
            obj.ptr.callMethod("startScheme");
        end

        function stopScheme()
            %
            % Stops the running scheme.
            %
            obj.ptr.callMethod("stopScheme");
        end
    end
end