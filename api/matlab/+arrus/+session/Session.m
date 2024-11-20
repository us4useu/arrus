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

        function state = getCurrentState(obj)
            % Returns a session current state (0-Stopped; 1-Started; 2-Closed)
            %
            res = obj.ptr.callMethod("getCurrentState", 1);
            state = res{1, 1};
        end

        function [buffer, frameOffsets, numberOfFrames, us4oems, frames, channels, rxOffset] = upload(obj, scheme)
            % Uploads a given scheme on the available devices.
            % Currently, the scheme upload is performed on the Us4R:0 device only.
            % After uploading a new sequence the previously returned output buffers will be in invalid state.
            %
            % :param scheme: scheme to upload (arrus.ops.us4r.Scheme)
            % :return: upload result information: output data buffer, metadata describing the data that will be generated
            res = obj.ptr.callMethod("upload", 7, scheme); % Note: 7 == number of output arrays
            buffer = arrus.framework.Buffer(res{1, 1});
            % FCM
            %% NOTE! bellow we assume numbering starting from 0!
            %% Consider output data array with dimensions (nSamples*totalNTxRxs, 32)
            %% For example: (32, 4096*12), where 12 = 2 us4oems * 3 Tx/Rxs (to cover full RX aperture) * 2 TxRxs.
            %% Frame offset says where given us4oem data starts, for example:
            %%
            %% frameOffsets(0): number of frame produced by us4OEM:0, frameOffset(1): number of frame produce by us4OEM:1
            %% NOTE! Each us4oem may gather different number of RF frames (e.g. 2nd us4oem will be not used,
            %% if the RX aperture will be fully covered by the first module).
            %%
            %% numberOfFrames: how many frames a given us4OEM produces
            %% us4oems: an array with dimensions (number of logical channels, number of logical TxRxs),
            %%   value: us4OEM number (starting from 0) that execute given logical (channel, frame)
            %% frames: an array with dimensions (number of logical channels, number of logical TxRxs),
            %%   value: physical frame number with data for the given logical (channel, frame)
            %% channels: an array with dimensions (number of logical channels, number of logical TxRxs),
            %%   value: physical channel number with data for the given logical (channel, frame)
            %% A tuple [us4oems(i, j), frames(i, j), channels(i, j)] uniquely addresses where
            %%   the given frame and channel are located in the output raw RF data.
            %% For example, logical frame 2, channel 33 can be obtained in the following way:
            %% us4oem = us4oems(lChannel, lFrame);
            %% frame = frames(lChannel, lFrame);
            %% channel = channels(lChannel, lFrame);
            %% us4oemFirstFrame = frameOffsets(us4oem);
            %% value = data(channel, :, us4oemFirstFrame+frame); % assume we have resized
                                                                 % raw data to (nPhysicalFrames, nSamples, nPhysicalChannels)
            %% rxOffset: Rx offset between tx time = 0 and rx time = 0 moments, scalar, unit: clock cycles. 

            frameOffsets = res{1, 2};
            numberOfFrames = res{1, 3};
            us4oems = res{1, 4};
            frames = res{1, 5};
            channels = res{1, 6};
            rxOffset = res{1, 7};
        end

        function [buffer, frameOffsets, numberOfFrames, us4oems, frames, channels, rxOffset] = setSubsequence(obj, start, stop, sri)
            % Sets the current TX/RX sequence to the [start, stop] subsequence (both ends inclusive).
            % 
            % This method requires that:
            % - start <= stop (when start == stop, the system will run a single TX/RX sequence),
            % - the scheme was uploaded,
            % - the TX/RX sequence length is greater than the `stop` value,
            % - the scheme is stopped.
            % 
            % :param start: the TX/RX number which should now be the first TX/RX
            % :param stop: the TX/RX number which should now be the last TX/RX
            % :param sri: the new SRI to apply [s], optional
            % :return: the new data buffer and metadata

            % Note: 7 == number of output arrays
            res = obj.ptr.callMethod("setSubsequence", 7, start, stop, sri);
            buffer = arrus.framework.Buffer(res{1, 1});
            frameOffsets = res{1, 2};
            numberOfFrames = res{1, 3};
            us4oems = res{1, 4};
            frames = res{1, 5};
            channels = res{1, 6};
            rxOffset = res{1, 7};
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
