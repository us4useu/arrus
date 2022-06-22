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
        
        function device = getDevice(obj, deviceId)
            % Returns a device with given identifier
            %
            % :param deviceId: a string pointing to some device, e.g. to get first available us4R device, use "/Us4R:0"
            res = obj.ptr.callMethod("getDevice", 1, deviceId);
            device = res{1, 1};
        end

        function buffer = upload(obj, scheme)
            % Uploads a given scheme on the available devices.
            % Currently, the scheme upload is performed on the Us4R:0 device only.
            % After uploading a new sequence the previously returned output buffers will be in invalid state.
            %
            % :param scheme: scheme to upload (arrus.ops.us4r.Scheme)
            % :return: upload result information: output data buffer, metadata describing the data that will be generated
            res = obj.ptr.callMethod("upload", 2, scheme);
            buffer = arrus.framework.Buffer(res{1, 1});
            % TODO Convert raw metadata to Matlab metadata
        end

        function run(obj)
            %
            % Runs the uploaded scheme.
            %
            % The behaviour of this method depends on the uploaded work mode:
            % - MANUAL: triggers execution of batch of sequences only ONCE,
            % - HOST, ASYNC: triggers execution of batch of sequences IN A LOOP (Host: trigger is on buffer element release).
            %   The run function can be called only once (before the scheme is stopped).
            obj.ptr.callMethod("run", 0);
        end

        function startScheme(obj)
            %
            % Starts the uploaded scheme.
            %
            obj.ptr.callMethod("startScheme", 0);
        end

        function stopScheme(obj)
            %
            % Stops the running scheme.
            %
            obj.ptr.callMethod("stopScheme", 0);
        end

        function close(obj)
            %
            % Stops all executors and closes connection with all devices.
            obj.ptr.callMethod("close", 0);
        end
    end
end